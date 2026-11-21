"""Compare conductor blocks with reusable shell FFTs and explicit shell-pair accumulation."""

from itertools import product
from time import perf_counter

import numpy as np

from multiplicative_convolution import ModularProduct


class ShellPairs:
    """Keep one FFT per input shell, then gather pullback frequencies for each pair."""

    def __init__(self, product_plan):
        self.plan = product_plan
        self.axes = []
        for local in product_plan.local:
            slots = [indices for _, _, indices in local.shells] + [np.array([local.power])]
            by_character = [dict(zip(chars, indices))
                            for chars, indices in zip(local.shell_characters, slots)]
            pairs = {}
            for a, b in product(range(local.power + 1), repeat=2):
                s = min(a + b, local.power)
                ids = local.shell_characters[s]
                pairs[a, b] = (
                    np.array([by_character[a][chi] for chi in ids]),
                    np.array([by_character[b][chi] for chi in ids]),
                    slots[s],
                )
            self.axes.append(pairs)

    def multiply(self, f, g):
        left = self.plan.transform(f, split_saturation=False)
        right = self.plan.transform(g, split_saturation=False)
        out = np.zeros_like(left)
        for pairs in product(*(axis.values() for axis in self.axes)):
            left_slots, right_slots, output_slots = zip(*pairs)
            out[np.ix_(*output_slots)] += left[np.ix_(*left_slots)] * right[np.ix_(*right_slots)]
        return self.plan.inverse(out, split_saturation=False)


if __name__ == "__main__":
    rng = np.random.default_rng(20261120)
    print("     n  extra setup ms  blocks ms  shell pairs ms  max disagreement")
    for n in (1024, 2187, 5040, 11520, 30030):
        plan = ModularProduct(n)
        start = perf_counter()
        shell_pairs = ShellPairs(plan)
        setup = perf_counter() - start
        comparisons = []
        for _ in range(5):
            f, g = rng.integers(-7, 8, (2, n))
            start = perf_counter()
            result = plan.multiply(f, g)
            block_time = perf_counter() - start
            start = perf_counter()
            expected = shell_pairs.multiply(f, g)
            shell_time = perf_counter() - start
            error = np.max(np.abs(result - expected))
            assert error < 1e-6, (n, error)
            comparisons.append((block_time, shell_time, error))
        timings = np.array(comparisons)
        blocks, shell = 1e3 * np.median(timings[:, :2], axis=0)
        print(f"{n:6}  {1e3*setup:14.3f}  {blocks:9.3f}  {shell:14.3f}  {timings[:, 2].max():.2e}")
