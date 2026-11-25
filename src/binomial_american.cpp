#include "experiment.hpp"

using namespace pricing;

int main(int argc, char **argv) {
    return run_experiment(argc, argv, "American option: CRR tree with early exercise at every node.",
        [](const Config &config) {
            if (config.exercise_dates && config.steps % config.exercise_dates != 0)
                throw std::invalid_argument("Tree steps must be divisible by exercise dates.");
            Row row;
            row.method = "binomial_american";
            row.steps = config.steps;
            row.reference_steps = config.reference_steps;
            row.estimate.price = timed([&] {
                return binomial_price(config.contract, config.steps, 1);
            }, row.runtime_ms);
            row.benchmark = config.reference_steps == config.steps ? row.estimate.price
                : binomial_price(config.contract, config.reference_steps, 1);
            std::vector<Row> rows{row};
            if (config.exercise_dates) {
                Row matched = row;
                matched.method = "binomial_american_grid";
                matched.estimate.price = timed([&] {
                    return binomial_price(config.contract, config.steps, config.steps / config.exercise_dates);
                }, matched.runtime_ms);
                rows.push_back(matched);
            }
            return rows;
        });
}
