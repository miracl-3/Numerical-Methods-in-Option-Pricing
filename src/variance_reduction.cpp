#include "experiment.hpp"

using namespace pricing;

// All three techniques estimate European payoffs only.
// Each method uses config.paths terminal payoff evaluations in its main sample.
// Independent control-variate and importance-sampling pilots count as extra work.
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
            const double log_weight = -shift * shifted_z + 0.5 * shift * shift;
            observations.add(payoff == 0.0 ? 0.0 : discount * payoff * std::exp(log_weight));
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

struct PilotObservation {
    double z;
    double discounted_payoff;
};

struct Objective {
    double second_moment;
    double gradient;
    double hessian;
};

// M_hat(mu) = mean[h(Z)^2 exp(mu^2/2 - mu*Z)], with pilot Z ~ N(0,1).
// Evaluate the objective and its two derivatives directly, as in the original procedure.
Objective pilot_objective(const std::vector<PilotObservation> &pilot,
                          std::size_t pilot_count, double shift) {
    Objective result{};
    for (const auto &observation : pilot) {
        const double difference = shift - observation.z;
        const double weight = observation.discounted_payoff * observation.discounted_payoff *
                              std::exp(0.5 * shift * shift - shift * observation.z);
        result.second_moment += weight;
        result.gradient += difference * weight;
        result.hessian += (1.0 + difference * difference) * weight;
    }
    const double count = static_cast<double>(pilot_count);
    result.second_moment /= count;
    result.gradient /= count;
    result.hessian /= count;
    if (!std::isfinite(result.second_moment) || !std::isfinite(result.gradient) ||
        !std::isfinite(result.hessian))
        throw std::overflow_error("Direct Newton objective exceeded the floating-point range.");
    if (result.second_moment <= 0.0 || result.hessian <= 0.0)
        throw std::runtime_error("Direct Newton requires a positive objective and second derivative.");
    return result;
}

struct NewtonStep {
    double shift;
    Objective objective;
    double step;
    bool step_limited;
};

struct ShiftFit {
    std::vector<NewtonStep> trace;
    std::size_t positive_payoffs = 0;
    std::string status = "deterministic";
    double shift = 0.0;
};

ShiftFit fit_importance_shift(const Config &config) {
    const auto &c = config.contract;
    ShiftFit fit;
    if (c.volatility == 0.0 || c.maturity == 0.0)
        return fit;
    // Fit on a separate stream, then freeze the shift before main-sample valuation.
    std::mt19937_64 generator(stream_seed(config.seed, 3));
    std::normal_distribution<double> normal;
    std::vector<PilotObservation> pilot;
    pilot.reserve(config.training_paths);
    for (std::size_t i = 0; i < config.training_paths; ++i) {
        const double z = normal(generator);
        const double stock = terminal_stock(c, z);
        if (!std::isfinite(stock)) throw std::overflow_error("Non-finite pilot stock price.");
        const double payoff = c.payoff(stock);
        if (payoff > 0.0)
            pilot.push_back({z, std::exp(-c.rate * c.maturity) * payoff});
    }
    fit.positive_payoffs = pilot.size();
    if (pilot.empty())
        throw std::runtime_error("Importance-sampling pilot contains no positive payoffs; "
                                 "increase --train-paths or use a less extreme contract.");

    // Start at the standardized exercise threshold, as in the original call code.
    // The put uses the corresponding lower-tail initialization.
    const double threshold = (std::log(c.strike / c.spot) -
        (c.rate - 0.5 * c.volatility * c.volatility) * c.maturity) /
        (c.volatility * std::sqrt(c.maturity));
    double shift = c.type == OptionType::Call ? std::max(0.0, threshold) : std::min(0.0, threshold);
    Objective objective = pilot_objective(pilot, config.training_paths, shift);
    fit.trace.push_back({shift, objective, 0.0, false});
    constexpr std::size_t max_iterations = 50;
    constexpr double tolerance = 1e-8;
    for (std::size_t iteration = 0; iteration <= max_iterations; ++iteration) {
        if (std::abs(objective.gradient) < tolerance) {
            fit.shift = shift;
            fit.status = "converged_gradient";
            return fit;
        }
        if (iteration == max_iterations)
            throw std::runtime_error("Importance-sampling Newton solver did not converge.");
        const double raw_step = objective.gradient / objective.hessian;
        // Preserve the original simple safeguard: limit each Newton correction to [-1,1].
        const double correction = std::clamp(raw_step, -1.0, 1.0);
        const double next_shift = shift - correction;
        const double actual_step = next_shift - shift;
        objective = pilot_objective(pilot, config.training_paths, next_shift);
        shift = next_shift;
        fit.trace.push_back({shift, objective, actual_step, correction != raw_step});
        if (std::abs(actual_step) < tolerance) {
            fit.shift = shift;
            fit.status = "converged_step";
            return fit;
        }
    }
    throw std::logic_error("Unreachable Newton solver state.");
}

void write_newton_trace(const Config &config, const ShiftFit &fit) {
    if (config.newton_trace.empty()) return;
    if (!config.output.empty() && std::filesystem::absolute(config.newton_trace).lexically_normal() ==
                                 std::filesystem::absolute(config.output).lexically_normal())
        throw std::invalid_argument("--newton-trace and --output must be different files.");
    const std::filesystem::path path(config.newton_trace);
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Cannot open Newton trace: " + config.newton_trace);
    output << "option_type,spot,strike,rate,volatility,maturity,training_paths,seed,"
              "positive_payoffs,iteration,shift,second_moment,gradient,hessian,step,step_limited,status\n";
    output << std::setprecision(17);
    const auto &c = config.contract;
    for (std::size_t i = 0; i < fit.trace.size(); ++i) {
        const auto &step = fit.trace[i];
        output << (c.type == OptionType::Call ? "call" : "put") << ',' << c.spot << ','
            << c.strike << ',' << c.rate << ',' << c.volatility << ',' << c.maturity << ','
            << config.training_paths << ',' << config.seed << ',' << fit.positive_payoffs << ','
            << i << ',' << step.shift << ',' << step.objective.second_moment << ','
            << step.objective.gradient << ',' << step.objective.hessian << ',' << step.step
            << ',' << step.step_limited
            << ',' << (i + 1 == fit.trace.size() ? fit.status : "iterating") << '\n';
    }
    output.flush();
    if (!output) throw std::runtime_error("Cannot write Newton trace: " + config.newton_trace);
}

int main(int argc, char **argv) {
    return run_experiment(argc, argv,
        "European variance reduction: antithetic, control variate, basic Newton importance sampling.",
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
                const bool importance = method == "importance_sampling";
                const bool needs_pilot = (control || importance) && config.contract.volatility > 0.0 &&
                                         config.contract.maturity > 0.0;
                row.training_paths = needs_pilot ? config.training_paths : 0;
                row.payoff_evaluations = config.paths + row.training_paths;
                ShiftFit fit;
                // Both pilot fitting and Newton iterations belong inside measured runtime.
                row.estimate = timed([&] {
                    row.control_beta = control ? fit_control_beta(config) : 0.0;
                    if (importance) {
                        fit = fit_importance_shift(config);
                        row.importance_shift = fit.shift;
                        row.newton_status = fit.status;
                        if (!fit.trace.empty()) {
                            row.newton_iterations = fit.trace.size() - 1;
                            row.newton_gradient = fit.trace.back().objective.gradient;
                            row.newton_second_moment_initial = fit.trace.front().objective.second_moment;
                            row.newton_second_moment_final = fit.trace.back().objective.second_moment;
                        }
                    }
                    return estimate_method(config, method, row.control_beta, row.importance_shift);
                }, row.runtime_ms);
                if (importance) write_newton_trace(config, fit);
                rows.push_back(row);
            }
            return rows;
        });
}
