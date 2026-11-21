#include "exact_product.hpp"

#include <charconv>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <string_view>

using namespace modular_product;

namespace {

std::string decimal(Wide magnitude) {
    std::string text;
    do {
        text += static_cast<char>('0' + magnitude % 10);
        magnitude /= 10;
    } while (magnitude != 0);
    std::reverse(text.begin(), text.end());
    return text;
}

std::string decimal(SignedWide value) {
    const Wide magnitude = static_cast<Wide>(value);
    return value < 0 ? "-" + decimal(Wide{0} - magnitude) : decimal(magnitude);
}

std::vector<SignedWide> direct_product(const std::vector<std::int32_t>& f, const std::vector<std::int32_t>& g) {
    const auto n = f.size();
    std::vector<SignedWide> result(n);
    for (std::size_t x = 0; x < n; ++x) {
        for (std::size_t y = 0; y < n; ++y) result[x * y % n] += SignedWide{f[x]} * g[y];
    }
    return result;
}

std::vector<std::int32_t> weights(Word n, std::mt19937& rng, std::int32_t limit) {
    std::vector<std::int32_t> values(n);
    std::uniform_int_distribution<std::int32_t> distribution(-limit, limit);
    for (auto& value : values) value = distribution(rng);
    return values;
}

std::vector<std::vector<Term>> make_baseline(const ProductPlan& plan) {
    std::vector<std::vector<Term>> terms;
    for (const auto& axis : plan.axes) terms.push_back(shell_pair_terms(axis));
    return terms;
}

void check(bool passed, const std::string& context) {
    if (!passed) throw std::runtime_error("failed: " + context);
}

void check_exact(Word n, std::mt19937& rng) {
    const ProductPlan plan(n);
    const auto baseline = make_baseline(plan);
    const ExactConvolution convolution(plan);
    auto f = weights(n, rng, std::numeric_limits<std::int32_t>::max());
    auto g = weights(n, rng, std::numeric_limits<std::int32_t>::max());
    f[0] = std::numeric_limits<std::int32_t>::min();
    g[0] = std::numeric_limits<std::int32_t>::max();
    const auto expected = direct_product(f, g);
    for (Word x = 0; x < n; ++x) check(plan.input_gcd[x] == std::gcd(x, n), "CRT gcd weights");
    const auto bound = coefficient_bound(plan, f, g);
    check(std::ranges::all_of(expected, [&](const auto x) {
        return static_cast<Wide>(x < 0 ? -x : x) <= bound;
    }), "signed coefficient bound, n=" + std::to_string(n));
    auto mod_result = convolution.multiply(f, g);
    check(mod_result == convolution.multiply(f, g, baseline), "modular implementations, n=" + std::to_string(n));
    const auto prime = convolution.field().modulus();
    for (Word i = 0; i < n; ++i) {
        const SignedWide remainder = expected[i] % prime;
        check(mod_result[i] == static_cast<Word>(remainder < 0 ? remainder + prime : remainder),
              "modular enumeration, n=" + std::to_string(n));
    }
    check(integer_product(plan, f, g) == expected, "signed CRT, n=" + std::to_string(n));

    std::fill(f.begin(), f.end(), 1);
    std::fill(g.begin(), g.end(), 1);
    const auto all_ones = integer_product(plan, f, g);
    check(static_cast<Wide>(all_ones[0]) == plan.largest_fiber, "largest fiber, n=" + std::to_string(n));
    check(std::ranges::all_of(all_ones, [&](const auto x) { return x <= all_ones[0]; }),
          "zero bin maximizes the fiber count, n=" + std::to_string(n));
    check(std::accumulate(all_ones.begin(), all_ones.end(), SignedWide{0}) == SignedWide{n} * n,
          "conservation of mass, n=" + std::to_string(n));
}

void self_test() {
    std::mt19937 rng(20261120);
    for (Word n = 2; n <= 256; ++n) check_exact(n, rng);
    for (Word n : {343u, 512u, 625u, 1296u, 2187u, 4914u, 5040u}) check_exact(n, rng);
    {
        const ProductPlan plan(72);
        std::vector<std::int32_t> zeros(72);
        auto g = weights(72, rng, 100);
        check(integer_product(plan, zeros, g) == std::vector<SignedWide>(72), "zero array");
        std::vector<std::int32_t> f(72);
        f[12] = -3;
        g.assign(72, 0);
        g[18] = 5;
        auto expected = std::vector<SignedWide>(72);
        expected[0] = -15;
        check(integer_product(plan, f, g) == expected, "zero divisor times zero divisor");
    }
    {
        constexpr Word n = 30030;
        const ProductPlan plan(n);
        const std::vector<std::int32_t> f(n, 2000000000);
        const std::vector<std::int32_t> g(n, -2000000000);
        const auto formula_factors = factor(n);
        const auto result = integer_product(plan, f, g);
        for (Word z = 0; z < n; ++z) {
            // For squarefree n, each local product has 2p-1 pairs at zero and p-1 otherwise.
            SignedWide count = 1;
            for (const auto& [p, _] : formula_factors) count *= z % p == 0 ? 2 * p - 1 : p - 1;
            check(result[z] == SignedWide{f[0]} * g[0] * count, "large signed CRT against fiber formula");
        }
    }
    std::cout << "Exact signed and field outputs agree with direct checks at 262 moduli and the "
                 "closed-form check at 30030.\n";
}

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return std::midpoint(values[(values.size() - 1) / 2], values[values.size() / 2]);
}

Wide operation_count(const ProductPlan& plan, std::span<const std::vector<Term>> baseline = {}) {
    Wide operations = 1;
    for (std::size_t i = 0; i < plan.axes.size(); ++i) {
        operations *= baseline.empty() ? plan.axes[i].block_terms.size() : baseline[i].size();
    }
    return operations;
}

std::vector<std::int64_t> direct_small_weight_product(std::span<const std::int32_t> f,
                                                     std::span<const std::int32_t> g) {
    const auto n = f.size();
    // At the benchmark's direct cutoff, even 49*n*n fits in int64_t.
    std::vector<std::int64_t> result(n);
    for (std::size_t x = 0; x < n; ++x) {
        const std::int64_t weight = f[x];
        std::size_t z = 0;
        for (std::size_t y = 0; y < n; ++y) {
            result[z] += weight * g[y];
            z += x;
            if (z >= n) z -= n;
        }
    }
    return result;
}

void benchmark() {
    constexpr int measured_trials = 16;
    constexpr Word direct_cutoff = 11520;
    std::mt19937 rng(20261120);
    std::cout << "n,prime,residue_and_field_setup_ms,blocks_ms,shell_pairs_ms,block_accumulation_ms,"
                 "shell_pair_accumulation_ms,integer_ms,direct_ms,block_products,shell_pair_products,one_prime_trials\n";
    std::cout << std::fixed << std::setprecision(6);
    for (Word n : {1024u, 2187u, 5040u, 11520u, 30030u, 45045u, 360360u, 510510u}) {
        const auto start = Clock::now();
        const ProductPlan plan(n);
        const ExactConvolution convolution(plan);
        const double setup = milliseconds(start);
        const auto baseline = make_baseline(plan);
        std::vector<double> block_times, baseline_times, block_accumulation_times, baseline_accumulation_times;
        std::vector<double> integer_times, direct_times;
        unsigned one_prime_trials = 0;
        const std::string modulus_label = ", n=" + std::to_string(n);
        for (int trial = 0; trial <= measured_trials; ++trial) {
            const auto f = weights(n, rng, 7);
            const auto g = weights(n, rng, 7);
            const auto evaluate = [&](bool shell_pairs) {
                double accumulation_ms = 0;
                const std::span<const std::vector<Term>> terms =
                    shell_pairs ? std::span<const std::vector<Term>>(baseline) : std::span<const std::vector<Term>>{};
                const auto begin = Clock::now();
                auto result = convolution.multiply(f, g, terms, &accumulation_ms);
                const auto elapsed = milliseconds(begin);
                if (trial != 0) {
                    (shell_pairs ? baseline_times : block_times).push_back(elapsed);
                    (shell_pairs ? baseline_accumulation_times : block_accumulation_times).push_back(accumulation_ms);
                }
                return result;
            };
            const auto first = evaluate(trial % 2 != 0);
            const auto second = evaluate(trial % 2 == 0);
            check(first == second, "benchmark field methods" + modulus_label);

            const auto evaluate_integer = [&] {
                const auto begin = Clock::now();
                auto result = integer_product(ProductPlan(n), f, g);
                const auto elapsed = milliseconds(begin);
                if (trial != 0) integer_times.push_back(elapsed);
                return result;
            };
            std::vector<SignedWide> exact;
            if (n <= direct_cutoff) {
                const auto evaluate_direct = [&] {
                    const auto begin = Clock::now();
                    auto result = direct_small_weight_product(f, g);
                    const auto elapsed = milliseconds(begin);
                    if (trial != 0) direct_times.push_back(elapsed);
                    return result;
                };
                std::vector<std::int64_t> direct;
                if (trial % 2 == 0) {
                    exact = evaluate_integer();
                    direct = evaluate_direct();
                } else {
                    direct = evaluate_direct();
                    exact = evaluate_integer();
                }
                check(std::ranges::equal(exact, direct), "benchmark direct comparison" + modulus_label);
            } else {
                exact = evaluate_integer();
            }
            const auto prime = convolution.field().modulus();
            if (trial != 0 && 2 * coefficient_bound(plan, f, g) < prime) ++one_prime_trials;
            bool exact_agrees = true;
            for (Word i = 0; i < n; ++i) {
                const auto remainder = exact[i] % prime;
                exact_agrees &= first[i] == static_cast<Word>(remainder < 0 ? remainder + prime : remainder);
            }
            check(exact_agrees, "benchmark CRT" + modulus_label);
        }
        std::cout << n << ',' << convolution.field().modulus() << ',' << setup << ',' << median(block_times) << ','
                  << median(baseline_times) << ',' << median(block_accumulation_times) << ','
                  << median(baseline_accumulation_times) << ',' << median(integer_times) << ',';
        if (!direct_times.empty()) std::cout << median(direct_times);
        std::cout << ',' << decimal(operation_count(plan)) << ',' << decimal(operation_count(plan, baseline)) << ','
                  << one_prime_trials << '\n';
    }
}

std::vector<std::int32_t> read_weights(Word n) {
    std::vector<std::int32_t> result(n);
    for (auto& value : result) {
        if (!(std::cin >> value)) throw std::invalid_argument("expected two dense vectors of signed 32-bit integer weights");
    }
    return result;
}

void convolve(std::string_view modulus_text) {
    Word modulus = 0;
    const char* end = modulus_text.data() + modulus_text.size();
    const auto [ptr, error] = std::from_chars(modulus_text.data(), end, modulus);
    if (error != std::errc{} || ptr != end || modulus < 2 || modulus >= (Word{1} << 30)) {
        throw std::invalid_argument("expected an integer modulus in [2, 2^30)");
    }
    const ProductPlan plan(modulus);
    const auto f = read_weights(plan.modulus);
    const auto g = read_weights(plan.modulus);
    if ((std::cin >> std::ws).peek() != std::char_traits<char>::eof()) {
        throw std::invalid_argument("unexpected input after the second weight vector");
    }
    const auto out = integer_product(plan, f, g);
    for (std::size_t i = 0; i < out.size(); ++i) std::cout << decimal(out[i]) << (i + 1 == out.size() ? '\n' : ' ');
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--self-test") self_test();
        else if (argc == 2 && std::string_view(argv[1]) == "--bench") benchmark();
        else if (argc == 3 && std::string_view(argv[1]) == "--convolve") convolve(argv[2]);
        else throw std::invalid_argument("usage: exact_product --self-test | --bench | --convolve MODULUS");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
