#include "experiment.hpp"

#include <array>
#include <numeric>

using namespace pricing;

struct Regression {
    std::array<double, 4> beta{};
    double center = 0.0;
    double scale = 1.0;
    bool valid = false;
    bool laguerre_basis = false;

    double continuation(double normalized_stock) const {
        if (laguerre_basis) {
            const double x = normalized_stock;
            const double weight = std::exp(-0.5 * x);
            return std::max(0.0, beta[0] + weight *
                (beta[1] + beta[2] * (1.0 - x) +
                 beta[3] * (1.0 - 2.0 * x + 0.5 * x * x)));
        }
        const double x = (normalized_stock - center) / scale;
        return std::max(0.0, beta[0] + x * (beta[1] + x * beta[2]));
    }
};

std::array<double, 4> basis_values(double normalized_stock, double center, double scale,
                                   bool laguerre_basis) {
    if (laguerre_basis) {
        const double x = normalized_stock;
        const double weight = std::exp(-0.5 * x);
        return {1.0, weight, weight * (1.0 - x),
                weight * (1.0 - 2.0 * x + 0.5 * x * x)};
    }
    const double x = (normalized_stock - center) / scale;
    return {1.0, x, x * x, 0.0};
}

// Reorthogonalized modified Gram-Schmidt QR avoids normal-equation instability.
Regression fit_continuation(const std::vector<double> &x, const std::vector<double> &y,
                            bool laguerre_basis) {
    Regression model;
    if (x.empty()) return model; // No in-the-money observations: continue.
    model.valid = true;
    model.laguerre_basis = laguerre_basis;
    Moments states;
    Moments targets;
    for (std::size_t i = 0; i < x.size(); ++i) {
        states.add(x[i]);
        targets.add(y[i]);
    }
    model.center = laguerre_basis ? 0.0 : states.mean;
    model.scale = laguerre_basis ? 1.0 : std::sqrt(states.variance());
    const std::size_t columns = laguerre_basis ? 4 : 3;
    if (x.size() < columns || (!laguerre_basis && model.scale < 1e-10)) {
        model.scale = 1.0;
        model.beta[0] = targets.mean;
        return model;
    }
    std::array<std::vector<double>, 4> q;
    std::array<std::array<double, 4>, 4> r{};
    std::array<double, 4> qty{};
    std::size_t rank = 0;
    for (std::size_t column = 0; column < columns; ++column) {
        q[column].resize(x.size());
        for (std::size_t i = 0; i < x.size(); ++i)
            q[column][i] = basis_values(x[i], model.center, model.scale, laguerre_basis)[column];
        for (int pass = 0; pass < 2; ++pass) {
            for (std::size_t previous = 0; previous < column; ++previous) {
                const double projection = std::inner_product(
                    q[previous].begin(), q[previous].end(), q[column].begin(), 0.0);
                r[previous][column] += projection;
                for (std::size_t i = 0; i < x.size(); ++i)
                    q[column][i] -= projection * q[previous][i];
            }
        }
        const double norm = std::sqrt(std::inner_product(
            q[column].begin(), q[column].end(), q[column].begin(), 0.0));
        if (norm < 1e-10 * std::sqrt(static_cast<double>(x.size()))) break;
        r[column][column] = norm;
        for (double &value : q[column]) value /= norm;
        qty[column] = std::inner_product(q[column].begin(), q[column].end(), y.begin(), 0.0);
        ++rank;
    }
    for (std::size_t i = rank; i-- > 0;) {
        double rhs = qty[i];
        for (std::size_t j = i + 1; j < rank; ++j) rhs -= r[i][j] * model.beta[j];
        model.beta[i] = rhs / r[i][i];
    }
    return model;
}

struct Policy {
    std::vector<Regression> models;
    bool exercise_at_zero = false;
};

struct TrainingResult {
    Policy policy;
    std::vector<double> discounted_cashflows;
    std::vector<std::size_t> stopping_times;
};

Paths simulate_antithetic_paths(const Contract &contract, std::size_t count,
                                std::size_t steps, std::uint64_t seed) {
    if (count < 4 || count % 2 != 0)
        throw std::invalid_argument("Antithetic LSM requires an even path count of at least four.");
    if (count > 50000000 / (steps + 1))
        throw std::invalid_argument("Path matrix exceeds the 50-million-value memory limit.");
    Paths paths(steps + 1, std::vector<double>(count, contract.spot));
    std::mt19937_64 generator(seed);
    std::normal_distribution<double> normal;
    const double dt = contract.maturity / static_cast<double>(steps);
    const double drift = (contract.rate - 0.5 * contract.volatility * contract.volatility) * dt;
    const double diffusion = contract.volatility * std::sqrt(dt);
    for (std::size_t time = 1; time <= steps; ++time) {
        for (std::size_t pair = 0; pair < count / 2; ++pair) {
            const double shock = diffusion * normal(generator);
            const std::size_t first = 2 * pair;
            const std::size_t second = first + 1;
            paths[time][first] = paths[time - 1][first] * std::exp(drift + shock);
            paths[time][second] = paths[time - 1][second] * std::exp(drift - shock);
            if (!std::isfinite(paths[time][first]) || !std::isfinite(paths[time][second]))
                throw std::overflow_error("Non-finite stock path.");
        }
    }
    return paths;
}

TrainingResult train_policy(const Config &config, const Paths &paths, bool laguerre_basis) {
    const auto &c = config.contract;
    TrainingResult result;
    Policy &policy = result.policy;
    policy.models.resize(config.steps + 1);
    if (c.type == OptionType::Call && c.rate >= 0.0) return result;
    const std::size_t path_count = paths.front().size();
    if (laguerre_basis) result.stopping_times.assign(path_count, config.steps);
    const double discount = std::exp(-c.rate * c.maturity / static_cast<double>(config.steps));
    std::vector<double> cashflows(path_count);
    for (std::size_t path = 0; path < path_count; ++path)
        cashflows[path] = c.payoff(paths.back()[path]);
    for (std::size_t time = config.steps; --time > 0;) {
        std::vector<double> states;
        std::vector<double> targets;
        states.reserve(path_count);
        targets.reserve(path_count);
        for (std::size_t path = 0; path < path_count; ++path) {
            // Cashflow always represents the selected future payoff discounted to this date.
            cashflows[path] *= discount;
            if (c.payoff(paths[time][path]) > 0.0) {
                states.push_back(paths[time][path] / c.strike);
                targets.push_back(cashflows[path]);
            }
        }
        const Regression model = fit_continuation(states, targets, laguerre_basis);
        policy.models[time] = model;
        if (!model.valid) continue;
        for (std::size_t path = 0; path < path_count; ++path) {
            const double exercise = c.payoff(paths[time][path]);
            if (exercise > 0.0 && exercise > model.continuation(paths[time][path] / c.strike)) {
                cashflows[path] = exercise;
                if (laguerre_basis) result.stopping_times[path] = time;
            }
        }
    }
    // Time-zero choice uses training only. Taking max(intrinsic, test mean) would bias valuation.
    const double continuation = discount * std::accumulate(cashflows.begin(), cashflows.end(), 0.0) /
                                static_cast<double>(path_count);
    policy.exercise_at_zero = c.payoff(c.spot) > continuation;
    if (laguerre_basis) {
        result.discounted_cashflows.resize(path_count);
        for (std::size_t path = 0; path < path_count; ++path)
            result.discounted_cashflows[path] = discount * cashflows[path];
        if (policy.exercise_at_zero) {
            std::fill(result.discounted_cashflows.begin(), result.discounted_cashflows.end(),
                      c.payoff(c.spot));
            std::fill(result.stopping_times.begin(), result.stopping_times.end(), 0);
        }
    }
    return result;
}

Estimate evaluate_policy(const Config &config, const Policy &policy, double &exercise_fraction) {
    const auto &c = config.contract;
    const double dt = c.maturity / static_cast<double>(config.steps);
    const double drift = (c.rate - 0.5 * c.volatility * c.volatility) * dt;
    const double diffusion = c.volatility * std::sqrt(dt);
    std::mt19937_64 generator(stream_seed(config.seed, 2)); // Never used in training.
    std::normal_distribution<double> normal;
    Moments payoffs;
    std::size_t early_exercises = 0;
    for (std::size_t path = 0; path < config.paths; ++path) {
        double stock = c.spot;
        std::size_t stopping_time = policy.exercise_at_zero ? 0 : config.steps;
        if (!policy.exercise_at_zero) {
            for (std::size_t time = 1; time <= config.steps; ++time) {
                stock *= std::exp(drift + diffusion * normal(generator));
                if (!std::isfinite(stock)) throw std::overflow_error("Non-finite stock path.");
                const auto &model = policy.models[time];
                const double exercise = c.payoff(stock);
                if (time < config.steps && model.valid && exercise > 0.0 &&
                    exercise > model.continuation(stock / c.strike)) {
                    stopping_time = time;
                    break;
                }
            }
        }
        early_exercises += stopping_time < config.steps;
        payoffs.add(std::exp(-c.rate * static_cast<double>(stopping_time) * dt) * c.payoff(stock));
    }
    exercise_fraction = static_cast<double>(early_exercises) / static_cast<double>(config.paths);
    return payoffs.estimate();
}

Estimate evaluate_in_sample_antithetic(const Config &config, const TrainingResult &training,
                                       double &exercise_fraction) {
    Moments pair_payoffs;
    std::size_t early_exercises = 0;
    for (std::size_t pair = 0; pair < config.paths / 2; ++pair) {
        const std::size_t first = 2 * pair;
        const std::size_t second = first + 1;
        pair_payoffs.add(0.5 * (training.discounted_cashflows[first] +
                                training.discounted_cashflows[second]));
        early_exercises += training.stopping_times[first] < config.steps;
        early_exercises += training.stopping_times[second] < config.steps;
    }
    exercise_fraction = static_cast<double>(early_exercises) / static_cast<double>(config.paths);
    return pair_payoffs.estimate();
}

int main(int argc, char **argv) {
    return run_experiment(argc, argv,
        "American option: Longstaff-Schwartz LSM; independent or paper Table 1 mode.",
        [](const Config &config) {
            if (config.lsm_mode == "paper" && config.contract.type != OptionType::Put)
                throw std::invalid_argument("Paper LSM mode reproduces the American put experiment only.");
            if (config.lsm_mode == "paper" && config.reference_steps % config.steps != 0)
                throw std::invalid_argument(
                    "Paper LSM reference steps must be divisible by the exercise-date count.");
            Row row;
            row.method = "lsm_american";
            row.steps = config.steps;
            row.paths = config.paths;
            const bool paper_mode = config.lsm_mode == "paper";
            const bool needs_training = config.contract.type != OptionType::Call ||
                                        config.contract.rate < 0.0;
            row.training_paths = paper_mode ? config.paths : (needs_training ? config.training_paths : 0);
            row.payoff_evaluations = paper_mode ? config.paths : config.paths + row.training_paths;
            row.reference_steps = config.reference_steps;
            const std::size_t exercise_every = paper_mode
                ? config.reference_steps / config.steps
                : 1;
            row.benchmark = binomial_price(config.contract, config.reference_steps, exercise_every);
            row.estimate = timed([&] {
                if (paper_mode) {
                    const Paths paths = simulate_antithetic_paths(
                        config.contract, config.paths, config.steps, stream_seed(config.seed, 1));
                    const TrainingResult training = train_policy(config, paths, true);
                    return evaluate_in_sample_antithetic(config, training, row.exercise_fraction);
                }
                const Paths paths = needs_training
                    ? simulate_paths(config.contract, config.training_paths, config.steps,
                                     stream_seed(config.seed, 1))
                    : Paths{};
                const TrainingResult training = train_policy(config, paths, false);
                return evaluate_policy(config, training.policy, row.exercise_fraction);
            }, row.runtime_ms);
            return std::vector<Row>{row};
        });
}
