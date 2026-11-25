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
    // expm1 avoids cancellation when dt is small (the reference uses 100,000
    // intervals). This is algebraically the ordinary CRR probability.
    const double p = std::expm1(c.rate * dt + log_up) / std::expm1(2.0 * log_up);
    if (!std::isfinite(p) || p < 0.0 || p > 1.0)
        throw std::invalid_argument("Invalid CRR probability; increase --steps.");
    if (std::log(c.spot) + static_cast<double>(steps) * log_up > 700.0)
        throw std::overflow_error("Tree stock values exceed the supported floating-point range.");
    const double discount = std::exp(-c.rate * dt);
    const double weight_down = discount * (1.0 - p);
    const double weight_up = discount * p;
    std::vector<double> values(steps + 1);
    // Every stock in a recombining tree is S0*exp(k*log_up), -N <= k <= N.
    // Precompute these payoffs once. Besides eliminating billions of stock
    // multiplications, evaluating each exponent independently prevents a zero
    // underflow at the lowest node from contaminating the rest of a level.
    std::vector<double> intrinsic(2 * steps + 1);
    const double log_spot = std::log(c.spot);
    for (std::size_t k = 0; k < intrinsic.size(); ++k) {
        const double exponent = log_spot +
            (static_cast<double>(k) - static_cast<double>(steps)) * log_up;
        intrinsic[k] = c.payoff(std::exp(exponent));
    }
    for (std::size_t j = 0; j <= steps; ++j)
        values[j] = intrinsic[2 * j];
    for (std::size_t i = steps; i-- > 0;) {
        const bool can_exercise = exercise_every != 0 && i % exercise_every == 0;
        if (can_exercise) {
            const double *payoffs = intrinsic.data() + steps - i;
            for (std::size_t j = 0; j <= i; ++j) {
                const double continuation = weight_down * values[j] + weight_up * values[j + 1];
                values[j] = std::max(payoffs[2 * j], continuation);
            }
        } else {
            for (std::size_t j = 0; j <= i; ++j)
                values[j] = weight_down * values[j] + weight_up * values[j + 1];
        }
    }
    if (!std::isfinite(values[0]))
        throw std::overflow_error("Non-finite tree price.");
    return values[0];
}

} // namespace pricing
