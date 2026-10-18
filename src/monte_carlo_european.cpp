#include "experiment.hpp"

using namespace pricing;

Estimate european_monte_carlo(const Config &config) {
    std::mt19937_64 generator(stream_seed(config.seed, 0));
    std::normal_distribution<double> normal;
    const auto &c = config.contract;
    const double discount = std::exp(-c.rate * c.maturity);
    Moments payoffs;
    // Exact GBM terminal sampling: no time-discretization bias for European payoffs.
    for (std::size_t i = 0; i < config.paths; ++i)
        payoffs.add(discount * c.payoff(terminal_stock(c, normal(generator))));
    return payoffs.estimate();
}

int main(int argc, char **argv) {
    return run_experiment(argc, argv, "European option: exact-terminal Monte Carlo and 95% CI.",
        [](const Config &config) {
            Row row;
            row.method = "monte_carlo_european";
            row.steps = 1;
            row.paths = config.paths;
            row.payoff_evaluations = config.paths;
            row.benchmark = black_scholes(config.contract);
            row.estimate = timed([&] { return european_monte_carlo(config); }, row.runtime_ms);
            return std::vector<Row>{row};
        });
}
