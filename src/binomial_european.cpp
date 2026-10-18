#include "experiment.hpp"

using namespace pricing;

int main(int argc, char **argv) {
    return run_experiment(argc, argv, "European option: Cox-Ross-Rubinstein binomial tree.",
        [](const Config &config) {
            Row row;
            row.method = "binomial_european";
            row.steps = config.steps;
            row.benchmark = black_scholes(config.contract);
            row.estimate.price = timed([&] {
                return binomial_price(config.contract, config.steps);
            }, row.runtime_ms);
            return std::vector<Row>{row};
        });
}
