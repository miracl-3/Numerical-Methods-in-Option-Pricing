#include "experiment.hpp"

using namespace pricing;

// A prescribed stopping rule is a feasible policy, not an optimal American solver.
// More paths remove sampling error, but do not remove exercise-policy bias.
bool exercise_now(const Contract &c, double stock, double boundary_ratio) {
    if (c.payoff(stock) <= 0.0)
        return false;
    if (c.type == OptionType::Call && c.rate >= 0.0)
        return false; // Non-dividend call: early exercise is never optimal when r >= 0.
    return c.type == OptionType::Put ? stock <= boundary_ratio * c.strike
                                    : stock >= boundary_ratio * c.strike;
}

Estimate fixed_policy_monte_carlo(const Config &config, double &exercise_fraction) {
    const auto &c = config.contract;
    const double dt = c.maturity / static_cast<double>(config.steps);
    const double drift = (c.rate - 0.5 * c.volatility * c.volatility) * dt;
    const double diffusion = c.volatility * std::sqrt(dt);
    std::mt19937_64 generator(stream_seed(config.seed, 0));
    std::normal_distribution<double> normal;
    Moments payoffs;
    std::size_t early_exercises = 0;
    for (std::size_t path = 0; path < config.paths; ++path) {
        double stock = c.spot;
        std::size_t time = 0;
        while (time < config.steps && !exercise_now(c, stock, config.boundary_ratio)) {
            stock *= std::exp(drift + diffusion * normal(generator));
            if (!std::isfinite(stock)) throw std::overflow_error("Non-finite stock path.");
            ++time;
        }
        early_exercises += time < config.steps;
        payoffs.add(std::exp(-c.rate * static_cast<double>(time) * dt) * c.payoff(stock));
    }
    exercise_fraction = static_cast<double>(early_exercises) / static_cast<double>(config.paths);
    return payoffs.estimate();
}

int main(int argc, char **argv) {
    return run_experiment(argc, argv,
        "American option: fixed exercise-policy Monte Carlo (lower bound in expectation).",
        [](const Config &config) {
            Row row;
            row.method = "monte_carlo_american_fixed_policy";
            row.steps = config.steps;
            row.paths = config.paths;
            row.payoff_evaluations = config.paths;
            row.reference_steps = config.reference_steps;
            row.benchmark = binomial_price(config.contract, config.reference_steps, 1);
            row.estimate = timed([&] {
                return fixed_policy_monte_carlo(config, row.exercise_fraction);
            }, row.runtime_ms);
            return std::vector<Row>{row};
        });
}
