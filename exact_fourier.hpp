#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace modular_product {

using Word = std::uint32_t;
using Wide = unsigned __int128;
using SignedWide = __int128;
using Factorization = std::vector<std::pair<Word, unsigned>>;
constexpr Word field_limit = (Word{1} << 31) - 1;

inline Factorization factor(Word n) {
    Factorization factors;
    for (Word p = 2; p <= n / p; p += p == 2 ? 1 : 2) {
        if (n % p != 0) continue;
        unsigned exponent = 0;
        do {
            n /= p;
            ++exponent;
        } while (n % p == 0);
        factors.emplace_back(p, exponent);
    }
    if (n > 1) factors.emplace_back(n, 1);
    return factors;
}

class Field {
    Word modulus_;
    std::uint64_t reciprocal_;

public:
    // Reduction works for moduli in [2, 2^31). Inversion requires a prime modulus.
    explicit Field(Word modulus)
        : modulus_(modulus), reciprocal_(std::numeric_limits<std::uint64_t>::max() / modulus + 1) {}

    Word modulus() const { return modulus_; }

    Word add(Word a, Word b) const {
        const Word total = a + b;
        return total >= modulus_ ? total - modulus_ : total;
    }

    Word sub(Word a, Word b) const { return a >= b ? a - b : a + (modulus_ - b); }

    // Safe for up to four products of reduced values, since the modulus is below 2^31.
    Word reduce(std::uint64_t value) const {
        const auto quotient = static_cast<std::uint64_t>((Wide{value} * reciprocal_) >> 64);
        const std::uint64_t multiple = quotient * modulus_;
        // The rounded-up reciprocal gives the exact quotient or one greater.
        return static_cast<Word>(value - multiple + (value < multiple ? modulus_ : 0));
    }

    Word mul(Word a, Word b) const { return reduce(std::uint64_t{a} * b); }

    Word pow(Word base, std::uint64_t exponent) const {
        Word result = 1;
        while (exponent != 0) {
            if (exponent & 1) result = mul(result, base);
            base = mul(base, base);
            exponent >>= 1;
        }
        return result;
    }

    Word inverse(Word a) const { return pow(a, modulus_ - 2); }

    Word from_signed(std::int32_t a) const {
        const std::int64_t remainder = std::int64_t{a} % modulus_;
        return static_cast<Word>(remainder < 0 ? remainder + modulus_ : remainder);
    }
};

inline Word primitive_root(Word prime) {
    if (prime == 2) return 1;
    const Field field(prime);
    const auto factors = factor(prime - 1);
    for (Word g = 2; g < prime; ++g) {
        const auto fails = [&](const auto& entry) {
            return field.pow(g, (prime - 1) / entry.first) == 1;
        };
        if (std::none_of(factors.begin(), factors.end(), fails)) return g;
    }
    throw std::invalid_argument("could not find a primitive root for the prime");
}

inline Word compatible_prime(std::uint64_t order, Word ceiling = field_limit) {
    if (order >= ceiling) throw std::invalid_argument("transform order exceeds the coefficient-prime range");
    for (std::uint64_t k = (ceiling - 1) / order; k > 0; --k) {
        const Word p = static_cast<Word>(k * order + 1);
        const auto factors = factor(p);
        if (factors.size() == 1 && factors.front() == std::pair{p, 1u}) return p;
    }
    throw std::invalid_argument("ran out of compatible coefficient primes below 2^31");
}

inline std::size_t radix_length(std::size_t n) {
    return std::has_single_bit(n) ? n : std::bit_ceil(2 * n - 1);
}

inline bool short_nonradix(std::size_t n) { return n <= 16 && !std::has_single_bit(n); }

class Radix2 {
    Field field_;
    Word inverse_length_;
    std::vector<std::size_t> reversed_;
    std::vector<Word> twiddles_;

public:
    Radix2(std::size_t n, Field field, Word generator)
        : field_(field), inverse_length_(field.inverse(static_cast<Word>(n))), reversed_(n), twiddles_(n) {
        if (!std::has_single_bit(n) || (field.modulus() - 1) % n != 0) {
            throw std::invalid_argument("radix-2 transform length does not divide p - 1");
        }
        for (std::size_t i = 1; i < n; ++i) reversed_[i] = (reversed_[i / 2] / 2) + (i % 2) * (n / 2);
        for (std::size_t half = 1; half < n; half *= 2) {
            const Word step = field_.pow(generator, (field_.modulus() - 1) / (2 * half));
            twiddles_[half] = 1;
            for (std::size_t j = 1; j < half; ++j) twiddles_[half + j] = field_.mul(twiddles_[half + j - 1], step);
        }
    }

    void apply(std::span<Word> values, bool inverse = false) const {
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (i < reversed_[i]) std::swap(values[i], values[reversed_[i]]);
        }
        for (std::size_t half = 1; half < values.size(); half *= 2) {
            for (std::size_t start = 0; start < values.size(); start += 2 * half) {
                for (std::size_t j = 0; j < half; ++j) {
                    const Word a = values[start + j];
                    const Word b = field_.mul(values[start + half + j], twiddles_[half + j]);
                    values[start + j] = field_.add(a, b);
                    values[start + half + j] = field_.sub(a, b);
                }
            }
        }
        if (inverse) {
            std::reverse(values.begin() + 1, values.end());
            for (auto& value : values) value = field_.mul(value, inverse_length_);
        }
    }
};

class Fourier {
    Field field_;
    Word inverse_length_;
    std::shared_ptr<const Radix2> radix_;
    std::vector<Word> small_weights_;
    std::vector<Word> chirp_;
    std::vector<Word> kernel_;

    void short_transform(std::span<Word> values, std::vector<Word>& work) const {
        const std::size_t n = values.size();
        work.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            Word sum = 0;
            const Word* weights = small_weights_.data() + k * n;
            std::size_t j = 0;
            for (; j + 4 <= n; j += 4) {
                const std::uint64_t block = std::uint64_t{values[j]} * weights[j]
                    + std::uint64_t{values[j + 1]} * weights[j + 1]
                    + std::uint64_t{values[j + 2]} * weights[j + 2]
                    + std::uint64_t{values[j + 3]} * weights[j + 3];
                sum = field_.add(sum, field_.reduce(block));
            }
            for (; j < n; ++j) sum = field_.add(sum, field_.mul(values[j], weights[j]));
            work[k] = sum;
        }
        std::copy(work.begin(), work.end(), values.begin());
    }

public:
    Fourier(std::size_t n, Field field, Word generator, std::shared_ptr<const Radix2> radix)
        : field_(field), inverse_length_(field.inverse(static_cast<Word>(n))), radix_(std::move(radix)) {
        if (std::has_single_bit(n)) return;
        if (short_nonradix(n)) {
            if ((field.modulus() - 1) % n != 0) throw std::invalid_argument("short transform requires a root of order n");
            small_weights_.resize(n * n);
            const Word root = field_.pow(generator, (field_.modulus() - 1) / n);
            Word row_step = 1;
            for (std::size_t k = 0; k < n; ++k) {
                Word value = 1;
                for (std::size_t j = 0; j < n; ++j) {
                    small_weights_[k * n + j] = value;
                    value = field_.mul(value, row_step);
                }
                row_step = field_.mul(row_step, root);
            }
            return;
        }
        if ((field.modulus() - 1) % (2 * n) != 0) {
            throw std::invalid_argument("Bluestein transform requires a root of order 2n");
        }
        chirp_.resize(n);
        kernel_.resize(radix_length(n));
        const Word root = field_.pow(generator, (field_.modulus() - 1) / (2 * n));
        const Word inverse_root = field_.inverse(root);
        const Word root_squared = field_.mul(root, root);
        const Word inverse_root_squared = field_.mul(inverse_root, inverse_root);
        Word chirp = 1;
        Word inverse_chirp = 1;
        Word step = root;
        Word inverse_step = inverse_root;
        kernel_[0] = 1;
        for (std::size_t j = 0; j < n; ++j) {
            chirp_[j] = chirp;
            if (j != 0) kernel_[j] = kernel_[kernel_.size() - j] = inverse_chirp;
            // (j+1)^2 - j^2 = 2j+1, so the chirp and its inverse need no per-bin exponentiation.
            chirp = field_.mul(chirp, step);
            inverse_chirp = field_.mul(inverse_chirp, inverse_step);
            step = field_.mul(step, root_squared);
            inverse_step = field_.mul(inverse_step, inverse_root_squared);
        }
        radix_->apply(kernel_);
    }

    void apply(std::span<Word> values, bool inverse, std::vector<Word>& work) const {
        if (small_weights_.empty() && chirp_.empty()) {
            radix_->apply(values, inverse);
            return;
        }
        if (!small_weights_.empty()) {
            short_transform(values, work);
        } else {
            work.assign(kernel_.size(), 0);
            for (std::size_t j = 0; j < values.size(); ++j) work[j] = field_.mul(values[j], chirp_[j]);
            radix_->apply(work);
            for (std::size_t j = 0; j < work.size(); ++j) work[j] = field_.mul(work[j], kernel_[j]);
            radix_->apply(work, true);
            for (std::size_t j = 0; j < values.size(); ++j) values[j] = field_.mul(work[j], chirp_[j]);
        }
        if (inverse) {
            std::reverse(values.begin() + 1, values.end());
            for (auto& value : values) value = field_.mul(value, inverse_length_);
        }
    }
};

}  // namespace modular_product
