"""Weighted modular-product convolution using unit characters and valuations."""

import argparse
from collections import defaultdict
from itertools import product
from time import perf_counter

import numpy as np


def factor(n):
    factors = []
    p = 2
    while p * p <= n:
        power = 0
        while n % p == 0:
            n //= p
            power += 1
        if power:
            factors.append((p, power))
        p = 3 if p == 2 else p + 2
    if n > 1:
        factors.append((n, 1))
    return factors


def odd_prime_generator(p, power):
    prime_divisors = [q for q, _ in factor(p - 1)]
    g = next(
        g for g in range(2, p)
        if all(pow(g, (p - 1) // q, p) != 1 for q in prime_divisors)
    )
    return g + p if power > 1 and pow(g, p - 1, p * p) == 1 else g


def powers(g, n, modulus):
    result = np.empty(n, dtype=np.intp)
    value = 1
    for i in range(n):
        result[i] = value
        value = value * g % modulus
    return result


class LocalTransform:
    """One prime power, with the trivial character's saturation split off."""

    def __init__(self, prime, power):
        self.modulus = prime**power
        self.power = power
        g = odd_prime_generator(prime, power) if prime != 2 else 5
        local_shells = []
        characters = defaultdict(list)
        for valuation in range(power):
            exponent = power - valuation
            ring = prime**exponent
            if prime != 2:
                order = ring - ring // prime
                units = powers(g, order, ring)
                group_shape = (order,)
                ids = np.arange(order) * prime**valuation
            elif exponent == 1:
                units = np.array([1])
                group_shape = (1,)
                ids = np.array([0])
            else:
                cyclic_size = 2**(exponent - 2)
                cyclic_units = powers(5, cyclic_size, ring)
                units = np.concatenate((cyclic_units, ring - cyclic_units))
                group_shape = (2, cyclic_size)
                base_ids = np.arange(cyclic_size) * 2**valuation
                ids = np.concatenate((base_ids, base_ids + 2**(power - 2)))
            for char in ids:
                characters[int(char)].append(valuation)
            local_shells.append((prime**valuation * units, group_shape, ids))

        # Character zero has p^a p^b = p^min(a+b,r) in valuation coordinates.
        # Its algebra maps to ordinary polynomials mod t^r and evaluation at 1.
        groups = defaultdict(list)
        groups[power].append(np.arange(power))
        groups[1].append(np.array([power]))
        positions = {(0, d): d for d in range(power)}
        next_slot = power + 1
        for char, degrees in sorted(characters.items()):
            if char == 0:
                continue
            length = len(degrees)
            groups[length].append(np.arange(next_slot, next_slot + length))
            for offset, degree in enumerate(degrees):
                positions[char, degree] = next_slot + offset
            next_slot += length
        assert next_slot == self.modulus
        self.length_groups = {length: np.stack(rows) for length, rows in groups.items()}
        self.shells = [
            (residues, shape, np.array([positions[int(char), degree] for char in ids]))
            for degree, (residues, shape, ids) in enumerate(local_shells)
        ]
        self.shell_characters = [ids for _, _, ids in local_shells] + [np.array([0])]

    def apply(self, tensor, axis, inverse=False, split_saturation=True):
        source = np.moveaxis(tensor, axis, 0)
        out = np.empty_like(source)
        tail = source.shape[1:]
        for residues, shape, slots in self.shells:
            axes = tuple(range(len(shape)))
            if inverse:
                grouped = source[slots].reshape(shape + tail)
                out[residues] = np.fft.ifftn(grouped, axes=axes).reshape((len(slots),) + tail)
            else:
                grouped = source[residues].reshape(shape + tail)
                out[slots] = np.fft.fftn(grouped, axes=axes).reshape((len(slots),) + tail)
        if inverse:
            out[0] = source[self.power]
            if split_saturation:
                out[0] -= source[:self.power].sum(axis=0)
        else:
            out[self.power] = source[0]
            if split_saturation:
                out[self.power] += out[:self.power].sum(axis=0)
        return np.moveaxis(out, 0, axis)


class ModularProduct:
    def __init__(self, modulus):
        if modulus < 2:
            raise ValueError("modulus must be at least 2")
        self.modulus = modulus
        self.local = [LocalTransform(prime, power) for prime, power in factor(modulus)]
        self.shape = tuple(transform.modulus for transform in self.local)
        residues = np.arange(modulus)
        self.crt_index = np.ravel_multi_index(tuple(residues % m for m in self.shape), self.shape)

    def transform(self, values, *, split_saturation=True):
        values = np.asarray(values)
        if values.shape != (self.modulus,):
            raise ValueError("expected one weight per residue")
        flat = np.empty(self.modulus, dtype=np.complex128)
        flat[self.crt_index] = values
        tensor = flat.reshape(self.shape)
        for axis, transform in enumerate(self.local):
            tensor = transform.apply(tensor, axis, split_saturation=split_saturation)
        return tensor

    def inverse(self, tensor, *, split_saturation=True):
        for axis, transform in reversed(list(enumerate(self.local))):
            tensor = transform.apply(tensor, axis, inverse=True,
                                     split_saturation=split_saturation)
        return tensor.ravel()[self.crt_index]

    def multiply(self, f, g, *, use_fft=False):
        left = self.transform(f)
        right = self.transform(g)
        out = np.empty_like(left)
        for charts in product(*(t.length_groups.items() for t in self.local)):
            lengths, slots = zip(*charts)
            index = np.ix_(*(s.ravel() for s in slots))
            if all(length == 1 for length in lengths):
                out[index] = left[index] * right[index]
                continue

            # Even axes batch blocks of the same degree, odd axes are polynomial degrees.
            batch_shape = tuple(size for s in slots for size in s.shape)
            left_group = left[index].reshape(batch_shape)
            right_group = right[index].reshape(batch_shape)
            mul = fft_blocks if use_fft else truncated_blocks
            coeffs = mul(left_group, right_group, lengths)
            out[index] = coeffs.reshape(tuple(s.size for s in slots))
        return self.inverse(out)

    def bilinear_sum(self, pairs):
        """Push independent weighted pairs through sum(x_j * y_j) modulo M."""
        spectrum = np.ones(self.modulus, dtype=np.complex128)
        for f, g in pairs:
            spectrum *= np.fft.fft(self.multiply(f, g))
        return np.fft.ifft(spectrum)


def fft_blocks(left, right, lengths):
    axes = tuple(2 * i + 1 for i, length in enumerate(lengths) if length > 1)
    padded = tuple(2 * left.shape[axis] - 1 for axis in axes)
    left_freq = np.fft.fftn(left, s=padded, axes=axes)
    right_freq = np.fft.fftn(right, s=padded, axes=axes)
    coeffs = np.fft.ifftn(left_freq * right_freq, axes=axes)
    truncate = tuple(slice(None) if i % 2 == 0 else slice(0, size)
                     for i, size in enumerate(left.shape))
    return coeffs[truncate]


def truncated_blocks(left, right, lengths):
    out = np.zeros_like(left)
    for degrees in product(*(range(length) for length in lengths)):
        coefficient = []
        prefix = []
        shifted = []
        for degree, length in zip(degrees, lengths):
            coefficient.extend((slice(None), slice(degree, degree + 1)))
            prefix.extend((slice(None), slice(0, length - degree)))
            shifted.extend((slice(None), slice(degree, length)))
        out[tuple(shifted)] += left[tuple(coefficient)] * right[tuple(prefix)]
    return out


def direct(f, g):
    n = len(f)
    residues = np.arange(n)
    out = np.zeros(n, dtype=np.complex128)
    for x in range(n):
        indices = (x * residues) % n
        weights = f[x] * g
        out.real += np.bincount(indices, weights=weights.real, minlength=n)
        if np.iscomplexobj(weights):
            out.imag += np.bincount(indices, weights=weights.imag, minlength=n)
    return out


def check(n, rng, complex_weights):
    f = rng.integers(-7, 8, n)
    g = rng.integers(-7, 8, n)
    if complex_weights:
        f = f + 1j * rng.integers(-7, 8, n)
        g = g + 1j * rng.integers(-7, 8, n)
    start = perf_counter()
    plan = ModularProduct(n)
    setup = perf_counter() - start
    start = perf_counter()
    result = plan.multiply(f, g)
    fast = perf_counter() - start
    start = perf_counter()
    fft_result = plan.multiply(f, g, use_fft=True)
    fft = perf_counter() - start
    start = perf_counter()
    expected = direct(f, g)
    brute = perf_counter() - start
    max_error = max(float(np.max(np.abs(result - expected))),
                    float(np.max(np.abs(fft_result - expected))))
    assert max_error < 1e-6, (n, max_error)
    return setup, fast, fft, brute, max_error


def check_bilinear_sum(rng):
    for n in (8, 9, 12, 15, 18):
        f, g, h, k = rng.integers(0, 5, (4, n))
        actual = ModularProduct(n).bilinear_sum([(f, g), (h, k)])
        expected = np.zeros(n, dtype=np.int64)
        for a, b, c, d in product(range(n), repeat=4):
            expected[(a * b + c * d) % n] += f[a] * g[b] * h[c] * k[d]
        assert np.max(np.abs(actual - expected)) < 1e-6, n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-small", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=3, help="repeats for the timing cases")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    rng = np.random.default_rng(20261120)
    errors = [check(n, rng, complex_weights=True)[-1] for n in range(2, args.max_small + 1)]
    if errors:
        print(f"Checked {len(errors)} small moduli with complex weights, max error {max(errors):.2e}")
    check_bilinear_sum(rng)
    print("Checked weighted two-block quadrics by direct four-variable enumeration")
    print(f"Timing medians over {args.repeat} input pairs")
    print("     n  setup ms  schoolbook ms  FFT ms  direct ms  one-shot speedup  max error")
    for n in (1024, 2187, 5040, 11520, 30030):
        metrics = np.array([check(n, rng, complex_weights=False) for _ in range(args.repeat)])
        setup, multiply, fft, brute = np.median(metrics[:, :4], axis=0)
        speedup = np.median(metrics[:, 3] / (metrics[:, 0] + metrics[:, 1]))
        error = metrics[:, 4].max()
        print(f"{n:6}  {1e3*setup:8.3f}  {1e3*multiply:13.3f}  {1e3*fft:6.3f}"
              f"  {1e3*brute:9.3f}  {speedup:15.1f}x  {error:.2e}")
