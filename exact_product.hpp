#pragma once

#include "exact_fourier.hpp"

#include <chrono>
#include <numeric>

namespace modular_product {

struct Shell {
    std::size_t cycle;
    std::vector<Word> residues;
    std::vector<Word> characters;
    std::vector<Word> slots;
};

struct Term {
    std::size_t left;
    std::size_t right;
    std::size_t output;
};

struct Axis {
    Word prime;
    Word modulus;
    std::size_t stride;
    std::vector<Shell> shells;
    std::vector<Term> block_terms;
};

inline std::vector<Term> shell_pair_terms(const Axis& axis) {
    const std::size_t r = axis.shells.size() - 1;
    std::vector<Term> terms;
    for (std::size_t a = 0; a <= r; ++a) {
        for (std::size_t b = 0; b <= r; ++b) {
            const std::size_t s = std::min(a + b, r);
            // A character's slots have consecutive indices on the shells where it exists.
            for (const auto slot : axis.shells[s].slots) {
                terms.push_back({(slot - s + a) * axis.stride, (slot - s + b) * axis.stride,
                                 slot * axis.stride});
            }
        }
    }
    return terms;
}

inline Axis make_axis(Word p, unsigned r, Word modulus, std::size_t stride) {
    Axis axis{p, modulus, stride, {}, {}};
    Word generator = p == 2 ? 5 : primitive_root(p);
    if (p != 2 && r > 1 && Field(p * p).pow(generator, p - 1) == 1) generator += p;

    const Word full_cycle = p == 2 ? std::max(Word{1}, modulus / 4) : modulus - modulus / p;
    std::vector<Word> lengths(modulus - modulus / p);
    for (Word local_modulus = modulus; local_modulus > 1; local_modulus /= p) {
        const Word scale = modulus / local_modulus;
        const Word cycle = p == 2 ? std::max(Word{1}, local_modulus / 4) : local_modulus - local_modulus / p;
        const bool signed_units = p == 2 && local_modulus > 2;
        auto& shell = axis.shells.emplace_back(Shell{cycle, {}, {}, {}});
        Word unit = 1;
        for (Word sign = 0; sign <= static_cast<Word>(signed_units); ++sign) {
            for (Word j = 0; j < cycle; ++j) {
                shell.residues.push_back(scale * (sign == 0 ? unit : local_modulus - unit));
                const Word character = sign * full_cycle + j * (full_cycle / cycle);
                shell.characters.push_back(character);
                ++lengths[character];
                unit = static_cast<Word>(std::uint64_t{unit} * generator % local_modulus);
            }
        }
    }
    axis.shells.push_back({1, {0}, {0}, {r}});

    // The trivial character occupies slots 0..r-1 plus the scalar at slot r.
    std::vector<Word> starts(lengths.size());
    Word next_slot = r + 1;
    for (std::size_t chi = 1; chi < starts.size(); ++chi) {
        starts[chi] = next_slot;
        next_slot += lengths[chi];
    }
    for (unsigned a = 0; a < r; ++a) {
        auto& shell = axis.shells[a];
        for (const Word chi : shell.characters) shell.slots.push_back(starts[chi] + a);
    }
    axis.block_terms.push_back({r * stride, r * stride, r * stride});
    for (std::size_t chi = 0; chi < starts.size(); ++chi) {
        for (Word a = 0; a < lengths[chi]; ++a) {
            for (Word b = 0; b < lengths[chi] - a; ++b) {
                const auto start = starts[chi];
                axis.block_terms.push_back({(start + a) * stride, (start + b) * stride,
                                            (start + a + b) * stride});
            }
        }
    }
    return axis;
}

class ProductPlan {
public:
    Word modulus;
    std::uint64_t root_order = 2;
    Wide largest_fiber = 1;
    std::vector<Axis> axes;
    std::vector<std::size_t> input_to_tensor;
    std::vector<Word> input_gcd;

    explicit ProductPlan(Word n) : modulus(n) {
        if (n < 2 || n >= (Word{1} << 30)) throw std::invalid_argument("expected 2 <= modulus < 2^30");
        std::size_t stride = n;
        for (const auto [p, r] : factor(n)) {
            Word power = 1;
            for (unsigned j = 0; j < r; ++j) power *= p;
            stride /= power;
            const Word cycle = p == 2 ? std::max(Word{1}, power / 4) : power - power / p;
            root_order = std::lcm(root_order, static_cast<std::uint64_t>(cycle));
            if (!short_nonradix(cycle)) {
                root_order = std::lcm(root_order, static_cast<std::uint64_t>(radix_length(cycle)));
                if (!std::has_single_bit(cycle)) root_order = std::lcm(root_order, std::uint64_t{2} * cycle);
            }
            if (root_order >= field_limit) throw std::invalid_argument("transform order exceeds the coefficient-prime range");
            axes.push_back(make_axis(p, r, power, stride));
            largest_fiber *= Wide{power} + Wide{r} * (power - power / p);
        }
        input_to_tensor.resize(n);
        input_gcd.assign(n, 1);
        for (const auto& axis : axes) {
            std::vector<Word> axis_gcd(axis.modulus);
            Word divisor = 1;
            for (const auto& shell : axis.shells) {
                for (const Word x : shell.residues) axis_gcd[x] = divisor;
                if (divisor < axis.modulus) divisor *= axis.prime;
            }
            for (Word x = 0; x < n; ++x) {
                const Word coordinate = x % axis.modulus;
                input_to_tensor[x] += coordinate * axis.stride;
                input_gcd[x] *= axis_gcd[coordinate];
            }
        }
    }
};

class ExactConvolution {
    const ProductPlan& plan_;
    Field field_;
    Word inverse_two_;
    std::map<std::size_t, Fourier> transforms_;

    void transform_axis(std::vector<Word>& input, std::vector<Word>& output, const Axis& axis,
                        bool inverse, bool split_saturation) const {
        std::vector<Word> values, work;
        const std::size_t r = axis.shells.size() - 1;
        const std::size_t stride = axis.stride;
        const std::size_t outer_stride = axis.modulus * stride;
        for (const auto& shell : axis.shells) {
            const auto& transform = transforms_.at(shell.cycle);
            const auto& read = inverse ? shell.slots : shell.residues;
            const auto& write = inverse ? shell.residues : shell.slots;
            values.resize(read.size());
            for (std::size_t start = 0; start < input.size(); start += outer_stride) {
                for (std::size_t i = start; i < start + stride; ++i) {
                    for (std::size_t j = 0; j < read.size(); ++j) values[j] = input[i + read[j] * stride];
                    auto low = std::span(values).first(shell.cycle);
                    transform.apply(low, inverse, work);
                    if (shell.cycle < values.size()) {
                        auto high = std::span(values).subspan(shell.cycle);
                        transform.apply(high, inverse, work);
                        for (std::size_t j = 0; j < shell.cycle; ++j) {
                            const Word sum = field_.add(low[j], high[j]);
                            const Word difference = field_.sub(low[j], high[j]);
                            low[j] = inverse ? field_.mul(inverse_two_, sum) : sum;
                            high[j] = inverse ? field_.mul(inverse_two_, difference) : difference;
                        }
                    }
                    for (std::size_t j = 0; j < write.size(); ++j) output[i + write[j] * stride] = values[j];
                }
            }
        }
        if (!split_saturation) return;
        for (std::size_t start = 0; start < input.size(); start += outer_stride) {
            for (std::size_t i = start; i < start + stride; ++i) {
                Word& scalar = output[i + (inverse ? 0 : r * stride)];
                for (std::size_t a = 0; a < r; ++a) {
                    scalar = inverse ? field_.sub(scalar, input[i + a * stride])
                                     : field_.add(scalar, output[i + a * stride]);
                }
            }
        }
    }

    void transform(std::vector<Word>& values, bool inverse, bool split_saturation) const {
        auto work = std::vector<Word>(values.size());
        for (const auto& axis : plan_.axes) {
            transform_axis(values, work, axis, inverse, split_saturation);
            values.swap(work);
        }
    }

    void accumulate(std::span<const std::span<const Term>> terms, std::size_t i, std::size_t j, std::size_t k,
                    std::span<const Word> left, std::span<const Word> right, std::span<Word> out) const {
        if (terms.size() == 1) {
            for (const auto [a, b, c] : terms.front()) {
                out[k + c] = field_.add(out[k + c], field_.mul(left[i + a], right[j + b]));
            }
            return;
        }
        for (const auto [a, b, c] : terms.front()) {
            accumulate(terms.subspan(1), i + a, j + b, k + c, left, right, out);
        }
    }

public:
    explicit ExactConvolution(const ProductPlan& plan, Word prime_ceiling = field_limit)
        : plan_(plan), field_(compatible_prime(plan.root_order, prime_ceiling)), inverse_two_(field_.inverse(2)) {
        const Word generator = primitive_root(field_.modulus());
        std::map<std::size_t, std::shared_ptr<const Radix2>> radix;
        for (const auto& axis : plan.axes) {
            for (const auto& shell : axis.shells) {
                const auto n = shell.cycle;
                if (transforms_.contains(n)) continue;
                std::shared_ptr<const Radix2> shared;
                if (!short_nonradix(n)) {
                    auto& cached = radix[radix_length(n)];
                    if (!cached) cached = std::make_shared<Radix2>(radix_length(n), field_, generator);
                    shared = cached;
                }
                transforms_.try_emplace(n, n, field_, generator, shared);
            }
        }
    }

    Field field() const { return field_; }

    std::vector<Word> multiply(std::span<const std::int32_t> f, std::span<const std::int32_t> g,
                               std::span<const std::vector<Term>> baseline = {},
                               double* accumulation_ms = nullptr) const {
        const auto n = plan_.modulus;
        if (f.size() != n || g.size() != n) throw std::invalid_argument("expected one weight per residue");
        if (!baseline.empty() && baseline.size() != plan_.axes.size()) {
            throw std::invalid_argument("expected one term list per CRT factor");
        }
        std::vector<Word> left(n), right(n), output(n);
        for (Word x = 0; x < n; ++x) {
            const std::size_t offset = plan_.input_to_tensor[x];
            left[offset] = field_.from_signed(f[x]);
            right[offset] = field_.from_signed(g[x]);
        }
        const bool split_saturation = baseline.empty();
        transform(left, false, split_saturation);
        transform(right, false, split_saturation);
        std::vector<std::span<const Term>> terms;
        if (split_saturation) {
            for (const auto& axis : plan_.axes) terms.emplace_back(axis.block_terms);
        } else {
            for (const auto& axis_terms : baseline) terms.emplace_back(axis_terms);
        }
        using Clock = std::chrono::steady_clock;
        const auto start = accumulation_ms ? Clock::now() : Clock::time_point{};
        accumulate(terms, 0, 0, 0, left, right, output);
        if (accumulation_ms) {
            *accumulation_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        }
        transform(output, true, split_saturation);
        for (Word x = 0; x < n; ++x) left[x] = output[plan_.input_to_tensor[x]];
        return left;
    }
};

inline Wide coefficient_bound(const ProductPlan& plan, std::span<const std::int32_t> f,
                              std::span<const std::int32_t> g) {
    struct Norms {
        Wide sum = 0;
        Wide maximum = 0;
        Wide gcd_sum = 0;
    };
    auto measures = [&](std::span<const std::int32_t> values) {
        Norms norms;
        for (Word x = 0; x < plan.modulus; ++x) {
            const auto v = values[x];
            const auto magnitude = static_cast<Wide>(v < 0 ? -std::int64_t{v} : v);
            norms.sum += magnitude;
            norms.maximum = std::max(norms.maximum, magnitude);
            norms.gcd_sum += magnitude * plan.input_gcd[x];
        }
        return norms;
    };
    const auto left = measures(f);
    const auto right = measures(g);
    return std::min({left.sum * right.sum, left.gcd_sum * right.maximum, right.gcd_sum * left.maximum});
}

inline std::vector<SignedWide> integer_product(const ProductPlan& plan, std::span<const std::int32_t> f,
                                              std::span<const std::int32_t> g) {
    if (f.size() != plan.modulus || g.size() != plan.modulus) throw std::invalid_argument("expected one weight per residue");
    const Wide twice_bound = 2 * coefficient_bound(plan, f, g);
    std::vector<Wide> values(plan.modulus);
    Wide modulus_product = 1;
    Word ceiling = field_limit;
    while (modulus_product <= twice_bound) {
        const ExactConvolution convolution(plan, ceiling);
        const Field field = convolution.field();
        const Word prime = field.modulus();
        const Word inverse = field.inverse(static_cast<Word>(modulus_product % prime));
        const auto residues = convolution.multiply(f, g);
        // The bound is below 2^123 for the accepted input sizes and coefficients.
        // Each CRT step can still exceed the 128-bit range if the prime is large.
        if (modulus_product > (~Wide{0}) / prime) throw std::overflow_error("CRT range exceeds 128 bits");
        for (Word x = 0; x < plan.modulus; ++x) {
            const Word diff = field.sub(residues[x], static_cast<Word>(values[x] % prime));
            values[x] += modulus_product * field.mul(diff, inverse);
        }
        modulus_product *= prime;
        ceiling = prime - 1;
    }
    std::vector<SignedWide> output;
    output.reserve(values.size());
    for (const Wide value : values) {
        output.push_back(value > modulus_product / 2 ? -static_cast<SignedWide>(modulus_product - value)
                                                     : static_cast<SignedWide>(value));
    }
    return output;
}

}  // namespace modular_product
