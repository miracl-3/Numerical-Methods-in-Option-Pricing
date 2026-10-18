#include "experiment.hpp"

using namespace pricing;

int main(int argc, char **argv) {
    return run_experiment(argc, argv, "American option: CRR tree with early exercise at every node.",
        [](const Config &config) {
            Row row;
            row.method = "binomial_american";
            row.steps = config.steps;
            row.reference_steps = config.reference_steps;
            row.benchmark = binomial_price(config.contract, config.reference_steps, 1);
            row.estimate.price = timed([&] {
                return binomial_price(config.contract, config.steps, 1);
            }, row.runtime_ms);
            return std::vector<Row>{row};
        });
}
