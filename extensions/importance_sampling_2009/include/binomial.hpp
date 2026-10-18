#pragma once

#include "model.hpp"

namespace pricing {

// CRR backward induction: O(steps^2) time and O(steps) working memory.
// exercise_every=0: European; =1: American grid; >1: Bermudan grid.
inline double binomial_price(const Contract &c, std::size_t steps,
                             std::size_t exercise_every = 0) {
    c.validate();
    if (steps == 0)
        throw std::invalid_argument("Tree steps must be positive.");
    if (c.maturity == 0.0)
        return c.payoff(c.spot);
    const double dt = c.maturity / static_cast<double>(steps);
    if (c.volatility == 0.0) {
        double value = black_scholes(c);
        if (exercise_every != 0) {
            for (std::size_t i = 0; i < steps; i += exercise_every) {
                const double t = static_cast<double>(i) * dt;
                value = std::max(value, std::exp(-c.rate * t) *
                                            c.payoff(c.spot * std::exp(c.rate * t)));
            }
        }
        return value;
    }
    const double log_up = c.volatility * std::sqrt(dt);
    const double up = std::exp(log_up);
    const double down = 1.0 / up;
    const double p = (std::exp(c.rate * dt) - down) / (up - down);
    if (!std::isfinite(p) || p < 0.0 || p > 1.0)
        throw std::invalid_argument("Invalid CRR probability; increase --steps.");
    if (std::log(c.spot) + static_cast<double>(steps) * log_up > 700.0)
        throw std::overflow_error("Tree stock values exceed the supported floating-point range.");
    const double discount = std::exp(-c.rate * dt);
    const double ratio = up * up;
    std::vector<double> values(steps + 1);
    double stock = c.spot * std::exp(-static_cast<double>(steps) * log_up);
    for (std::size_t j = 0; j <= steps; ++j, stock *= ratio)
        values[j] = c.payoff(stock);
    for (std::size_t i = steps; i-- > 0;) {
        stock = c.spot * std::exp(-static_cast<double>(i) * log_up);
        const bool can_exercise = exercise_every != 0 && i % exercise_every == 0;
        for (std::size_t j = 0; j <= i; ++j, stock *= ratio) {
            const double continuation = discount * ((1.0 - p) * values[j] + p * values[j + 1]);
            values[j] = can_exercise ? std::max(c.payoff(stock), continuation) : continuation;
        }
    }
    if (!std::isfinite(values[0]))
        throw std::overflow_error("Non-finite tree price.");
    return values[0];
}

} // namespace pricing
