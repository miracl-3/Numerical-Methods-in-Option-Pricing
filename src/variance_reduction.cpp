#include "experiment.hpp"

using namespace pricing;

// All three techniques estimate European payoffs only.
// Each method uses config.paths terminal payoff evaluations in its main sample.
// The independent control-variate pilot is recorded as additional work.
Estimate estimate_method(const Config &config, const std::string &method,
                         double beta, double shift) {
    const auto &c = config.contract;
    const double discount = std::exp(-c.rate * c.maturity);
    std::mt19937_64 generator(stream_seed(config.seed, 0));
    std::normal_distribution<double> normal;
    Moments observations;
    const std::size_t samples = method == "antithetic" ? config.paths / 2 : config.paths;
    for (std::size_t i = 0; i < samples; ++i) {
        const double z = normal(generator);
        if (method == "antithetic") {
            // Pair averages, rather than correlated individual payoffs, are independent.
            observations.add(0.5 * discount * (c.payoff(terminal_stock(c, z)) +
                                               c.payoff(terminal_stock(c, -z))));
        } else if (method == "control_variate") {
            const double stock = terminal_stock(c, z);
            // E[exp(-rT) S_T] = S_0 under q=0.
            observations.add(discount * c.payoff(stock) - beta * (discount * stock - c.spot));
        } else if (method == "importance_sampling") {
            const double shifted_z = z + shift;
            const double payoff = c.payoff(terminal_stock(c, shifted_z));
            // f(z)/g(z) = exp(-shift*z + shift^2/2), z ~ N(shift, 1).
            // Form the weighted payoff in log space to avoid 0 * infinity.
            const double log_weight = -shift * shifted_z + 0.5 * shift * shift;
            observations.add(shift == 0.0 ? discount * payoff : payoff == 0.0 ? 0.0 :
                std::exp(std::log(payoff) - c.rate * c.maturity + log_weight));
        } else {
            observations.add(discount * c.payoff(terminal_stock(c, z)));
        }
    }
    return observations.estimate();
}

double fit_control_beta(const Config &config) {
    const auto &c = config.contract;
    if (c.volatility == 0.0 || c.maturity == 0.0)
        return 0.0;
    std::mt19937_64 generator(stream_seed(config.seed, 1));
    std::normal_distribution<double> normal;
    Moments controls;
    Moments payoffs;
    double covariance_sum = 0.0;
    const double discount = std::exp(-c.rate * c.maturity);
    for (std::size_t i = 0; i < config.training_paths; ++i) {
        const double stock = terminal_stock(c, normal(generator));
        const double control = discount * stock;
        const double payoff = discount * c.payoff(stock);
        const double old_control_mean = controls.mean;
        controls.add(control);
        payoffs.add(payoff);
        covariance_sum += (control - old_control_mean) * (payoff - payoffs.mean);
    }
    return controls.m2 > 0.0 ? covariance_sum / controls.m2 : 0.0;
}

double importance_shift(const Contract &c) {
    if (c.volatility == 0.0 || c.maturity == 0.0)
        return 0.0;
    const double threshold = (std::log(c.strike / c.spot) -
        (c.rate - 0.5 * c.volatility * c.volatility) * c.maturity) /
        (c.volatility * std::sqrt(c.maturity));
    // A transparent heuristic, not an optimized or universally superior proposal.
    return c.type == OptionType::Call ? std::clamp(threshold + 0.5, 0.0, 4.0)
                                     : std::clamp(threshold - 0.5, -4.0, 0.0);
}

int main(int argc, char **argv) {
    return run_experiment(argc, argv,
        "European variance reduction: antithetic, control variate, importance sampling.",
        [](const Config &config) {
            if (config.paths % 2 != 0)
                throw std::invalid_argument("--paths must be even for the antithetic comparison.");
            std::vector<Row> rows;
            const double benchmark = black_scholes(config.contract);
            for (const std::string method : {"plain_mc", "antithetic", "control_variate",
                                              "importance_sampling"}) {
                Row row;
                row.method = method;
                row.paths = config.paths;
                row.steps = 1;
                row.benchmark = benchmark;
                const bool control = method == "control_variate";
                const bool needs_pilot = control && config.contract.volatility > 0.0 &&
                                         config.contract.maturity > 0.0;
                row.training_paths = needs_pilot ? config.training_paths : 0;
                row.payoff_evaluations = config.paths + row.training_paths;
                row.importance_shift = method == "importance_sampling" ?
                                           importance_shift(config.contract) : 0.0;
                // Pilot fitting belongs inside the control method's measured runtime.
                row.estimate = timed([&] {
                    row.control_beta = control ? fit_control_beta(config) : 0.0;
                    return estimate_method(config, method, row.control_beta, row.importance_shift);
                }, row.runtime_ms);
                rows.push_back(row);
            }
            return rows;
        });
}
