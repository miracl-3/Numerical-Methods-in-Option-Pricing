#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace pricing {

enum class OptionType { Call, Put };

struct Contract {
    double spot = 100.0;
    double strike = 100.0;
    double rate = 0.05;
    double volatility = 0.20;
    double maturity = 1.0;
    OptionType type = OptionType::Put;

    void validate() const {
        for (double value : {spot, strike, rate, volatility, maturity}) {
            if (!std::isfinite(value))
                throw std::invalid_argument("Contract parameters must be finite.");
        }
        if (spot <= 0.0 || strike <= 0.0 || volatility < 0.0 || maturity < 0.0)
            throw std::invalid_argument(
                "Spot/strike must be positive; volatility/maturity must be nonnegative.");
    }

    double payoff(double stock) const {
        return std::max(type == OptionType::Call ? stock - strike : strike - stock, 0.0);
    }
};

inline double normal_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

inline double black_scholes(const Contract &c) {
    c.validate();
    if (c.maturity == 0.0)
        return c.payoff(c.spot);
    if (c.volatility == 0.0)
        return std::exp(-c.rate * c.maturity) * c.payoff(c.spot * std::exp(c.rate * c.maturity));
    const double scale = c.volatility * std::sqrt(c.maturity);
    const double d1 = (std::log(c.spot / c.strike) +
                       (c.rate + 0.5 * c.volatility * c.volatility) * c.maturity) / scale;
    const double d2 = d1 - scale;
    const double strike_pv = c.strike * std::exp(-c.rate * c.maturity);
    return c.type == OptionType::Call
               ? c.spot * normal_cdf(d1) - strike_pv * normal_cdf(d2)
               : strike_pv * normal_cdf(-d2) - c.spot * normal_cdf(-d1);
}

inline double terminal_stock(const Contract &c, double z) {
    return c.spot * std::exp((c.rate - 0.5 * c.volatility * c.volatility) * c.maturity +
                             c.volatility * std::sqrt(c.maturity) * z);
}

// SplitMix64 creates deterministic, separate seed streams for training and evaluation.
inline std::uint64_t stream_seed(std::uint64_t seed, std::uint64_t stream) {
    std::uint64_t x = seed + 0x9e3779b97f4a7c15ULL * (stream + 1);
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct Estimate {
    double price = 0.0;
    double standard_error = 0.0;
    std::size_t samples = 0;
};

// Welford updates avoid cancellation in variance = E[X^2] - E[X]^2.
struct Moments {
    std::size_t count = 0;
    double mean = 0.0;
    double m2 = 0.0;

    void add(double value) {
        if (!std::isfinite(value))
            throw std::overflow_error("Non-finite simulated value; reduce parameter magnitudes.");
        ++count;
        const double delta = value - mean;
        mean += delta / static_cast<double>(count);
        m2 += delta * (value - mean);
    }

    double variance() const {
        return count > 1 ? std::max(0.0, m2 / static_cast<double>(count - 1)) : 0.0;
    }

    Estimate estimate() const {
        if (count < 2)
            throw std::invalid_argument("At least two independent samples are required.");
        return {mean, std::sqrt(variance() / static_cast<double>(count)), count};
    }
};

// Time-major storage makes each regression date contiguous in memory.
using Paths = std::vector<std::vector<double>>;

inline Paths simulate_paths(const Contract &c, std::size_t count, std::size_t steps,
                            std::uint64_t seed) {
    if (steps == 0 || count < 2)
        throw std::invalid_argument("Need positive steps and at least two paths.");
    if (count > 50000000 / (steps + 1))
        throw std::invalid_argument("Path matrix exceeds the 50-million-value memory limit.");
    Paths paths(steps + 1, std::vector<double>(count, c.spot));
    std::mt19937_64 generator(seed);
    std::normal_distribution<double> normal;
    const double dt = c.maturity / static_cast<double>(steps);
    const double drift = (c.rate - 0.5 * c.volatility * c.volatility) * dt;
    const double diffusion = c.volatility * std::sqrt(dt);
    for (std::size_t time = 1; time <= steps; ++time) {
        for (std::size_t path = 0; path < count; ++path) {
            paths[time][path] = paths[time - 1][path] * std::exp(drift + diffusion * normal(generator));
            if (!std::isfinite(paths[time][path]))
                throw std::overflow_error("Non-finite stock path.");
        }
    }
    return paths;
}

} // namespace pricing
