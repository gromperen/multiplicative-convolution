# Multiplicative convolution modulo M

Given two arrays `f` and `g` indexed by the residues modulo an integer `M ≥ 2`, compute

```
h[z] = Σ_{x·y ≡ z (mod M)} f[x] · g[y]        for z = 0, …, M−1.
```

If `f` and `g` are the distributions of independent random residues `X` and `Y`,
then `h` is the distribution of `X·Y mod M`. Direct enumeration costs `M²`
multiplications. This repository computes `h` in **O(M log M)** operations for
**any** modulus, including exact integer output.

The method is explained in the blog post
[Fast multiplicative convolution modulo a composite number](https://gromperen.github.io/blog/2026/fast-multiplicative-convolution/).

## How it works

1. **Prime powers.** For `M = p^r`, write each nonzero residue as `p^a · u` with
   `u` a unit modulo `p^(r−a)`. Each "shell" `a` is a copy of a unit group,
   so it gets an ordinary FFT over that group's characters.
2. **Blocks.** For a fixed character `χ` with conductor `p^c`, the coefficients
   on shells `0, …, r−c` form a polynomial in a formal variable `t`. Products of
   residues become products of these polynomials, truncated at degree `r − c`.
   A separate total-mass scalar recovers `h[0]`. In algebraic terms,

   ```
   C[(Z/p^r, ·)] ≅ C × C[t]/(t^r) × Π_{χ≠1} C[t]/(t^(r−c(χ)+1)).
   ```

3. **Composite moduli.** The Chinese remainder theorem splits `Z/M` into prime
   powers. The array is reshaped into one axis per prime power, the local
   transform is applied along every axis, blocks are multiplied as multivariate
   truncated polynomials, and the transforms are inverted.

The transforms cost O(M log M). The block products cost
O(M log log M), and exactly `M` when `M` is squarefree.

## Files

| File | Contents |
| --- | --- |
| `exact_fourier.hpp` | Number-theoretic transforms over prime fields (radix 2, small direct DFTs, Bluestein). |
| `exact_product.hpp` | The block transform and multiplication, the shell-pair baseline, and exact signed reconstruction by CRT. |
| `exact_product.cpp` | Command-line driver: self-test, benchmark, and a `--convolve` mode. |
| `multiplicative_convolution.py` | A NumPy prototype using floating-point FFTs. |
| `shell_pairs.py` | The shell-pair baseline in NumPy, for comparison with the prototype. |

## C++: exact integer arithmetic

Requires a C++20 compiler with `__int128` (Clang or GCC).

```sh
clang++ -O3 -std=c++20 -Wall -Wextra exact_product.cpp -o exact_product
./exact_product --self-test      # checks 262 moduli against direct enumeration
./exact_product --bench          # benchmark used in the blog post
```

`--convolve M` reads two whitespace-separated vectors of `M` signed 32-bit
integers, each in residue order `0, 1, …, M−1`, and prints `h` in the same order:

```sh
$ printf '%s\n' '0 3 2 0' '0 4 1 0' | ./exact_product --convolve 4
2 12 11 0
```

The computation runs over prime fields `F_P` with `P < 2^31` and `P − 1`
divisible by the needed transform lengths. It bounds the largest possible
`|h[z]|` from the inputs and uses enough primes to recover the signed result by
the Chinese remainder theorem.

## Python prototype

Requires NumPy.

```python
import numpy as np
from multiplicative_convolution import ModularProduct

f = np.random.randint(-7, 8, 12)
g = np.random.randint(-7, 8, 12)
h = ModularProduct(12).multiply(f, g)
```

```sh
python multiplicative_convolution.py --max-small 256 --repeat 3   # checks against direct enumeration, then times
python shell_pairs.py                                             # compares blocks with shell pairs
```

## Benchmarks

Single-threaded on an Apple M5 Max (Apple clang 21, `-O3`). Times are in ms,
written as blocks / shell pairs where both appear. Each figure is the median
over five processes of the per-process median over 16 random input pairs with
weights in `[−7, 7]`.

| M | Products | Multiply step | Full transform pass | Exact, end to end | Direct `M²` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 = 2¹⁰ | 2,037 / 2,102 | 0.0017 / 0.0018 | 0.048 / 0.048 | 0.078 | 0.97 |
| 5,040 = 2⁴·3²·5·7 | 10,395 / 41,328 | 0.0093 / 0.0365 | 0.301 / 0.312 | 0.361 | 23.32 |
| 11,520 = 2⁸·3²·5 | 27,665 / 61,264 | 0.0255 / 0.0602 | 0.829 / 0.848 | 0.901 | 122.06 |
| 360,360 = 2³·3²·5·7·11·13 | 660,660 / 4,127,760 | 0.581 / 3.556 | 27.93 / 29.47 | 30.41 | — |
| 510,510 = 2·3·5·7·11·13·17 | 510,510 / 4,668,300 | 0.447 / 3.927 | 53.63 / 54.16 | 57.57 | — |

"Shell pairs" transforms each shell once and multiplies every pair of shells
explicitly, as existing `2^n` implementations do. It has the same asymptotics.
The block method cuts the products by up to 9×, but transforms dominate the
runtime, so end-to-end gains over shell pairs are 0–6%.

## Related work

- J. A. Wood, *Factoring the semigroup determinant of a finite commutative chain
  ring*, in *Coding Theory, Cryptography and Related Areas*, Springer, 2000,
  pp. 249–259.
- B. Steinberg, *Factoring the Dedekind–Frobenius determinant of a semigroup*,
  J. Algebra 605 (2022) 1–36, Corollary 6.13.
  [arXiv:2108.02166](https://arxiv.org/abs/2108.02166)
- Prime moduli via discrete logarithms: Library Checker
  [`mul_modp_convolution`](https://judge.yosupo.jp/problem/mul_modp_convolution),
  [maspy's library](https://maspypy.github.io/library/mod/multiplicative_convolution_mod_p.hpp),
  [cp-algorithms](https://lib.cp-algorithms.com/verify/poly/convolution_mul.test.cpp).
- Powers of two:
  [suisen's library](https://suisen-cp.github.io/cp-library-cpp/library/convolution/multiplicative_convolution_mod_2%5En.hpp),
  [maspy's library](https://maspypy.github.io/library/mod/multiplicative_convolution_mod_2n.hpp).
