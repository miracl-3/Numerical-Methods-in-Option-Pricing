#include "experiment.hpp"

#include <array>
#include <numeric>

using namespace pricing;

struct Regression {
    std::array<double, 3> beta{};
    double center = 0.0;
    double scale = 1.0;
    bool valid = false;

    double continuation(double normalized_stock) const {
        const double x = (normalized_stock - center) / scale;
        return std::max(0.0, beta[0] + x * (beta[1] + x * beta[2]));
    }
};

// Quadratic least squares using reorthogonalized modified Gram-Schmidt QR.
// Centering/scaling and rank fallback avoid unstable normal equations on ITM paths.
Regression fit_continuation(const std::vector<double> &x, const std::vector<double> &y) {
    Regression model;
    if (x.empty()) return model; // No in-the-money observations: continue.
    model.valid = true;
    Moments states;
    Moments targets;
    for (std::size_t i = 0; i < x.size(); ++i) {
        states.add(x[i]);
        targets.add(y[i]);
    }
    model.center = states.mean;
    model.scale = std::sqrt(states.variance());
    if (x.size() < 3 || model.scale < 1e-10) {
        model.scale = 1.0;
        model.beta[0] = targets.mean;
        return model;
    }
    std::array<std::vector<double>, 3> q;
    std::array<std::array<double, 3>, 3> r{};
    std::array<double, 3> qty{};
    std::size_t rank = 0;
    for (std::size_t column = 0; column < 3; ++column) {
        q[column].resize(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double state = (x[i] - model.center) / model.scale;
            q[column][i] = column == 0 ? 1.0 : (column == 1 ? state : state * state);
        }
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

Policy train_policy(const Config &config) {
    const auto &c = config.contract;
    Policy policy;
    policy.models.resize(config.steps + 1);
    if (c.type == OptionType::Call && c.rate >= 0.0) return policy;
    const auto paths = simulate_paths(c, config.training_paths, config.steps,
                                      stream_seed(config.seed, 1));
    const double discount = std::exp(-c.rate * c.maturity / static_cast<double>(config.steps));
    std::vector<double> cashflows(config.training_paths);
    for (std::size_t path = 0; path < config.training_paths; ++path)
        cashflows[path] = c.payoff(paths.back()[path]);
    for (std::size_t time = config.steps; --time > 0;) {
        std::vector<double> states;
        std::vector<double> targets;
        states.reserve(config.training_paths);
        targets.reserve(config.training_paths);
        for (std::size_t path = 0; path < config.training_paths; ++path) {
            // Cashflow always represents the selected future payoff discounted to this date.
            cashflows[path] *= discount;
            if (c.payoff(paths[time][path]) > 0.0) {
                states.push_back(paths[time][path] / c.strike);
                targets.push_back(cashflows[path]);
            }
        }
        const Regression model = fit_continuation(states, targets);
        policy.models[time] = model;
        if (!model.valid) continue;
        for (std::size_t path = 0; path < config.training_paths; ++path) {
            const double exercise = c.payoff(paths[time][path]);
            if (exercise > 0.0 && exercise > model.continuation(paths[time][path] / c.strike))
                cashflows[path] = exercise;
        }
    }
    // Time-zero choice uses training only. Taking max(intrinsic, test mean) would bias valuation.
    const double continuation = discount * std::accumulate(cashflows.begin(), cashflows.end(), 0.0) /
                                static_cast<double>(config.training_paths);
    policy.exercise_at_zero = c.payoff(c.spot) > continuation;
    return policy;
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

int main(int argc, char **argv) {
    return run_experiment(argc, argv,
        "American option: Longstaff-Schwartz LSM, independently trained/evaluated stopping rule.",
        [](const Config &config) {
            Row row;
            row.method = "lsm_american";
            row.steps = config.steps;
            row.paths = config.paths;
            const bool needs_training = config.contract.type != OptionType::Call ||
                                        config.contract.rate < 0.0;
            row.training_paths = needs_training ? config.training_paths : 0;
            row.payoff_evaluations = config.paths + row.training_paths;
            row.reference_steps = config.reference_steps;
            row.benchmark = binomial_price(config.contract, config.reference_steps, 1);
            row.estimate = timed([&] {
                const Policy policy = train_policy(config);
                return evaluate_policy(config, policy, row.exercise_fraction);
            }, row.runtime_ms);
            return std::vector<Row>{row};
        });
}
