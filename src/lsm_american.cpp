#include "experiment.hpp"

#include <array>
#include <numeric>

using namespace pricing;

enum class Basis { Laguerre, Quadratic }; // Quadratic is for the paper's eight-path example.

double normalized_payoff(const Contract &contract, double stock) {
    return contract.payoff(stock) / contract.strike;
}

std::array<double, 4> basis_values(double stock_over_strike, double center, double scale,
                                  Basis basis) {
    if (basis == Basis::Laguerre) {
        const double x = stock_over_strike;
        const double weight = std::exp(-0.5 * x);
        return {1.0, weight, weight * (1.0 - x),
                weight * (1.0 - 2.0 * x + 0.5 * x * x)};
    }
    const double x = (stock_over_strike - center) / scale;
    return {1.0, x, x * x, 0.0};
}

struct Regression {
    std::array<double, 4> beta{};
    double center = 0.0;
    double scale = 1.0;
    Basis basis = Basis::Laguerre;
    bool valid = false;
    std::size_t itm_samples = 0;
    std::size_t rank = 0;
    double condition_inf = 0.0; // ||R||_inf ||R^-1||_inf for retained columns.
    double diagonal_min = 0.0;
    double diagonal_max = 0.0;
    std::string status = "no_itm";

    double continuation(double stock_over_strike) const {
        const auto values = basis_values(stock_over_strike, center, scale, basis);
        return std::max(0.0, std::inner_product(beta.begin(), beta.end(), values.begin(), 0.0));
    }
};

using SmallMatrix = std::array<std::array<double, 4>, 4>;

double triangular_condition_inf(const SmallMatrix &r, std::size_t rank) {
    SmallMatrix inverse{};
    for (std::size_t column = 0; column < rank; ++column) {
        for (std::size_t i = rank; i-- > 0;) {
            double rhs = i == column ? 1.0 : 0.0;
            for (std::size_t j = i + 1; j < rank; ++j) rhs -= r[i][j] * inverse[j][column];
            inverse[i][column] = rhs / r[i][i];
        }
    }
    double norm = 0.0, inverse_norm = 0.0;
    for (std::size_t i = 0; i < rank; ++i) {
        double row_sum = 0.0, inverse_row_sum = 0.0;
        for (std::size_t j = 0; j < rank; ++j) {
            row_sum += std::abs(r[i][j]);
            inverse_row_sum += std::abs(inverse[i][j]);
        }
        norm = std::max(norm, row_sum);
        inverse_norm = std::max(inverse_norm, inverse_row_sum);
    }
    return norm * inverse_norm;
}

// Reorthogonalized modified Gram-Schmidt QR; no normal equations or dependencies.
Regression fit_continuation(const std::vector<double> &x, const std::vector<double> &y,
                            Basis basis = Basis::Laguerre) {
    if (x.size() != y.size()) throw std::invalid_argument("Regression sizes differ.");
    Regression model;
    model.basis = basis;
    model.itm_samples = x.size();
    if (x.empty()) return model;
    model.valid = true;
    Moments states, targets;
    for (std::size_t i = 0; i < x.size(); ++i) { states.add(x[i]); targets.add(y[i]); }
    model.center = basis == Basis::Laguerre ? 0.0 : states.mean;
    model.scale = basis == Basis::Laguerre ? 1.0 : std::sqrt(states.variance());
    const std::size_t columns = basis == Basis::Laguerre ? 4 : 3;
    if (x.size() < columns || model.scale < 1e-10) {
        model.scale = 1.0;
        model.beta[0] = targets.mean;
        model.rank = 1;
        model.condition_inf = 1.0;
        model.diagonal_min = model.diagonal_max = std::sqrt(static_cast<double>(x.size()));
        model.status = "constant_fallback";
        return model;
    }
    std::array<std::vector<double>, 4> q;
    SmallMatrix r{};
    std::array<double, 4> qty{};
    for (std::size_t column = 0; column < columns; ++column) q[column].resize(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const auto values = basis_values(x[i], model.center, model.scale, basis);
        for (std::size_t column = 0; column < columns; ++column) q[column][i] = values[column];
    }
    const double rank_tolerance = 1e-10 * std::sqrt(static_cast<double>(x.size()));
    for (std::size_t column = 0; column < columns; ++column) {
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
        if (norm < rank_tolerance) break;
        r[column][column] = norm;
        for (double &value : q[column]) value /= norm;
        qty[column] = std::inner_product(q[column].begin(), q[column].end(), y.begin(), 0.0);
        ++model.rank;
    }
    for (std::size_t i = model.rank; i-- > 0;) {
        double rhs = qty[i];
        for (std::size_t j = i + 1; j < model.rank; ++j) rhs -= r[i][j] * model.beta[j];
        model.beta[i] = rhs / r[i][i];
    }
    model.diagonal_min = model.diagonal_max = r[0][0];
    for (std::size_t i = 1; i < model.rank; ++i) {
        model.diagonal_min = std::min(model.diagonal_min, r[i][i]);
        model.diagonal_max = std::max(model.diagonal_max, r[i][i]);
    }
    model.condition_inf = triangular_condition_inf(r, model.rank);
    model.status = model.rank < columns ? "rank_deficient" : "full_rank";
    return model;
}

bool should_exercise(const Contract &contract, const Regression &model, double stock) {
    const double exercise = normalized_payoff(contract, stock);
    return model.valid && exercise > 0.0 && exercise > model.continuation(stock / contract.strike);
}

struct Policy {
    std::vector<Regression> models;
    bool exercise_at_zero = false;
    double normalized_time_zero_continuation = 0.0;
};

struct Cashflows {
    std::vector<double> discounted_cashflows; // Dollar amounts at time zero.
    std::vector<std::size_t> stopping_times;
};

struct TrainingResult {
    Policy policy;
    Cashflows valuation;
};

void write_trace_header(std::ostream &out) {
    out << "time_index,path,stock,regression_target_normalized,continuation_normalized,"
           "exercise_normalized,exercise_now,cashflow_after_normalized,stopping_time_after\n";
    out << std::setprecision(17);
}

TrainingResult train_policy(const Config &config, const Paths &paths,
                             Basis basis = Basis::Laguerre, std::ostream *trace = nullptr) {
    const auto &c = config.contract;
    if (config.steps == 0 || paths.size() != config.steps + 1 || paths.front().empty())
        throw std::invalid_argument("Path matrix does not match exercise dates.");
    const std::size_t count = paths.front().size();
    for (const auto &level : paths)
        if (level.size() != count) throw std::invalid_argument("Ragged path matrix.");
    TrainingResult result;
    auto &policy = result.policy;
    auto &valuation = result.valuation;
    policy.models.resize(config.steps + 1);
    valuation.stopping_times.assign(count, config.steps);
    const double discount = std::exp(-c.rate * c.maturity / static_cast<double>(config.steps));
    std::vector<double> cashflows(count);
    for (std::size_t path = 0; path < count; ++path)
        cashflows[path] = normalized_payoff(c, paths.back()[path]);
    const bool hold_call = c.type == OptionType::Call && c.rate >= 0.0;
    for (std::size_t time = config.steps; --time > 0;) {
        std::vector<double> states, targets;
        states.reserve(count); targets.reserve(count);
        for (std::size_t path = 0; path < count; ++path) {
            cashflows[path] *= discount;
            if (normalized_payoff(c, paths[time][path]) > 0.0) {
                states.push_back(paths[time][path] / c.strike);
                targets.push_back(cashflows[path]);
            }
        }
        auto &model = policy.models[time];
        if (hold_call) {
            model.status = "optimal_hold_call";
            model.itm_samples = states.size();
        }
        else model = fit_continuation(states, targets, basis);
        for (std::size_t path = 0; path < count; ++path) {
            const double stock = paths[time][path];
            const double target = cashflows[path];
            const bool exercise = should_exercise(c, model, stock);
            if (exercise) {
                cashflows[path] = normalized_payoff(c, stock);
                valuation.stopping_times[path] = time;
            }
            if (trace)
                *trace << time << ',' << path + 1 << ',' << stock << ',' << target << ','
                       << model.continuation(stock / c.strike) << ',' << normalized_payoff(c, stock)
                       << ',' << exercise << ',' << cashflows[path] << ','
                       << valuation.stopping_times[path] << '\n';
        }
    }
    policy.normalized_time_zero_continuation = discount *
        std::accumulate(cashflows.begin(), cashflows.end(), 0.0) / static_cast<double>(count);
    // Exactly the same t=0 exercise opportunity as the CRR reference.
    // For out-of-sample pricing this choice uses training data only.
    policy.exercise_at_zero = normalized_payoff(c, c.spot) > policy.normalized_time_zero_continuation;
    valuation.discounted_cashflows.resize(count);
    for (std::size_t path = 0; path < count; ++path) {
        const double target = discount * cashflows[path];
        const double selected = policy.exercise_at_zero ? normalized_payoff(c, c.spot) : target;
        valuation.discounted_cashflows[path] = c.strike * selected;
        if (policy.exercise_at_zero || c.maturity == 0.0) valuation.stopping_times[path] = 0;
        if (trace)
            *trace << 0 << ',' << path + 1 << ',' << c.spot << ',' << target << ','
                   << policy.normalized_time_zero_continuation << ',' << normalized_payoff(c, c.spot)
                   << ',' << policy.exercise_at_zero << ',' << selected << ','
                   << valuation.stopping_times[path] << '\n';
    }
    return result;
}

Cashflows evaluate_policy(const Config &config, const Policy &policy, const Paths &paths) {
    const auto &c = config.contract;
    if (config.steps == 0 || paths.size() != config.steps + 1 || paths.front().empty() ||
        policy.models.size() != config.steps + 1)
        throw std::invalid_argument("Evaluation exercise grid differs from the fitted policy.");
    const std::size_t count = paths.front().size();
    for (const auto &level : paths)
        if (level.size() != count) throw std::invalid_argument("Ragged evaluation matrix.");
    const double dt = c.maturity / static_cast<double>(config.steps);
    Cashflows result;
    result.discounted_cashflows.resize(count);
    result.stopping_times.assign(count, config.steps);
    for (std::size_t path = 0; path < count; ++path) {
        std::size_t time = config.steps;
        if (policy.exercise_at_zero || c.maturity == 0.0) time = 0;
        else {
            for (std::size_t date = 1; date < config.steps; ++date) {
                if (should_exercise(c, policy.models[date], paths[date][path])) { time = date; break; }
            }
        }
        result.stopping_times[path] = time;
        result.discounted_cashflows[path] = std::exp(-c.rate * dt * static_cast<double>(time)) *
                                             c.payoff(paths[time][path]);
    }
    return result;
}

Estimate summarize_cashflows(const Config &config, const Paths &paths, const Cashflows &valuation,
                              PathSampling sampling, Row &row) {
    const auto count = valuation.discounted_cashflows.size();
    const std::size_t group_size = sampling == PathSampling::Antithetic ? 2 : 1;
    if (count < 2 * group_size || count % group_size != 0)
        throw std::invalid_argument("Need at least two complete sampling groups for an SE.");
    Moments grouped_payoffs, path_payoffs, grouped_european, path_european, grouped_premiums, path_premiums;
    std::size_t early_exercises = 0;
    const double discount = std::exp(-config.contract.rate * config.contract.maturity);
    for (std::size_t first = 0; first < count; first += group_size) {
        double american_sum = 0.0, european_sum = 0.0;
        for (std::size_t offset = 0; offset < group_size; ++offset) {
            const auto path = first + offset;
            const double american = valuation.discounted_cashflows[path];
            const double european = discount * config.contract.payoff(paths.back()[path]);
            american_sum += american; european_sum += european;
            path_payoffs.add(american); path_european.add(european);
            path_premiums.add(american - european);
            early_exercises += config.contract.maturity > 0.0 && valuation.stopping_times[path] < config.steps;
        }
        grouped_payoffs.add(american_sum / static_cast<double>(group_size));
        grouped_european.add(european_sum / static_cast<double>(group_size));
        grouped_premiums.add((american_sum - european_sum) / static_cast<double>(group_size));
    }
    row.exercise_fraction = static_cast<double>(early_exercises) / static_cast<double>(count);
    row.naive_path_standard_error = path_payoffs.estimate().standard_error;
    row.european_price = grouped_european.mean;
    row.european_standard_error = grouped_european.estimate().standard_error;
    row.european_naive_path_standard_error = path_european.estimate().standard_error;
    // Same-path European premiums are diagnostics only; no control-variate adjustment.
    row.early_exercise_premium = grouped_premiums.mean;
    row.early_exercise_premium_standard_error = grouped_premiums.estimate().standard_error;
    row.early_exercise_premium_naive_path_standard_error = path_premiums.estimate().standard_error;
    return grouped_payoffs.estimate();
}

void add_regression_summary(const Policy &policy, Row &row) {
    row.exercise_at_zero = policy.exercise_at_zero;
    std::size_t fitted_dates = 0;
    for (std::size_t time = 1; time + 1 < policy.models.size(); ++time) {
        const auto &model = policy.models[time];
        if (model.status == "optimal_hold_call") continue;
        if (fitted_dates++ == 0) {
            row.regression_itm_min = model.itm_samples;
            row.regression_rank_min = model.rank;
        }
        row.regression_itm_min = std::min(row.regression_itm_min, model.itm_samples);
        row.regression_itm_max = std::max(row.regression_itm_max, model.itm_samples);
        row.regression_rank_min = std::min(row.regression_rank_min, model.rank);
        row.regression_rank_max = std::max(row.regression_rank_max, model.rank);
        row.regression_condition_inf_max = std::max(row.regression_condition_inf_max, model.condition_inf);
        row.regression_no_itm_dates += model.status == "no_itm";
        row.regression_fallback_dates += model.valid && model.status != "full_rank";
    }
}

void write_regressions(std::ostream &out, const Config &config, const Policy &policy) {
    out << "time_index,time,training_paths,itm_samples,rank,condition_inf_retained_R,"
           "qr_diagonal_min,qr_diagonal_max,status,beta0,beta1,beta2,beta3,center,scale\n";
    out << std::setprecision(17);
    for (std::size_t time = 1; time < config.steps; ++time) {
        const auto &model = policy.models[time];
        out << time << ',' << config.contract.maturity * static_cast<double>(time) /
            static_cast<double>(config.steps) << ',' << config.training_paths << ',' << model.itm_samples
            << ',' << model.rank << ',' << model.condition_inf << ',' << model.diagonal_min
            << ',' << model.diagonal_max << ',' << model.status;
        for (const auto beta : model.beta) out << ',' << beta;
        out << ',' << model.center << ',' << model.scale << '\n';
    }
}

void open_csv(std::ofstream &out, const std::string &name) {
    if (name.empty()) return;
    const std::filesystem::path path(name);
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    out.open(path);
    if (!out) throw std::runtime_error("Cannot open CSV: " + name);
}

std::vector<Row> lsm_experiment(Config config) {
    if (config.lsm_mode == "independent") {
        if (!config.sampling_explicit) config.sampling = "plain";
        if (!config.valuation_explicit) config.valuation = "out-of-sample";
    }
    if (config.sampling != "plain" && config.sampling != "antithetic")
        throw std::invalid_argument("--sampling must be plain or antithetic.");
    if (config.valuation != "in-sample" && config.valuation != "out-of-sample" && config.valuation != "both")
        throw std::invalid_argument("--valuation must be in-sample, out-of-sample, or both.");
    const bool antithetic = config.sampling == "antithetic";
    const PathSampling sampling = antithetic ? PathSampling::Antithetic : PathSampling::Plain;
    const bool in_sample = config.valuation != "out-of-sample";
    const bool out_of_sample = config.valuation != "in-sample";
    if (config.valuation == "in-sample") config.training_paths = config.paths;
    if (antithetic && (config.paths % 2 || config.training_paths % 2))
        throw std::invalid_argument("Antithetic LSM requires even --paths and --train-paths counts.");
    if (config.reference_steps % config.steps != 0)
        throw std::invalid_argument("LSM reference steps must be divisible by the exercise-date count.");
    if (!config.lsm_trace.empty() && config.training_paths > 1000)
        throw std::invalid_argument("Use at most 1000 training paths for --lsm-trace.");
    std::vector<std::filesystem::path> destinations;
    for (const auto &name : {config.output, config.regression_diagnostics, config.lsm_trace}) {
        if (name.empty()) continue;
        const auto path = std::filesystem::absolute(name).lexically_normal();
        if (std::find(destinations.begin(), destinations.end(), path) != destinations.end())
            throw std::invalid_argument("Pricing, diagnostics, and trace must be different files.");
        destinations.push_back(path);
    }
    std::ofstream diagnostics, trace;
    open_csv(diagnostics, config.regression_diagnostics); open_csv(trace, config.lsm_trace);
    if (trace.is_open()) write_trace_header(trace);
    Row base;
    base.method = "lsm_american";
    base.steps = config.steps;
    base.training_paths = config.training_paths;
    base.training_seed = stream_seed(config.seed, 1);
    base.sampling_method = config.sampling;
    base.basis = "weighted_laguerre_4";
    base.training_antithetic_pairs = antithetic ? config.training_paths / 2 : 0;
    base.reference_steps = config.reference_steps;
    base.benchmark = timed([&] {
        const double grid = binomial_price(config.contract, config.reference_steps,
                                           config.reference_steps / config.steps);
        base.american_reference_crr = binomial_price(config.contract, config.reference_steps, 1);
        return grid;
    }, base.reference_runtime_ms);
    TrainingResult training;
    Row training_row = base;
    {
        // Release this matrix before creating any independent evaluation matrix.
        double elapsed = 0.0;
        training = timed([&] {
            const Paths paths = simulate_paths(config.contract, config.training_paths, config.steps,
                                                base.training_seed, sampling);
            auto fitted = train_policy(config, paths, Basis::Laguerre, trace.is_open() ? &trace : nullptr);
            if (in_sample) training_row.estimate = summarize_cashflows(config, paths, fitted.valuation, sampling, training_row);
            return fitted;
        }, elapsed);
        base.training_runtime_ms = elapsed;
    }
    if (diagnostics.is_open()) write_regressions(diagnostics, config, training.policy);
    if ((diagnostics.is_open() && !diagnostics) || (trace.is_open() && !trace))
        throw std::runtime_error("Failed to write LSM diagnostics.");
    base.time_zero_continuation = config.contract.strike * training.policy.normalized_time_zero_continuation;
    add_regression_summary(training.policy, base);
    std::vector<Row> rows;
    if (in_sample) {
        // Preserve the cashflow diagnostics collected while the training paths existed.
        training_row.training_runtime_ms = base.training_runtime_ms;
        training_row.runtime_ms = base.training_runtime_ms;
        training_row.valuation_mode = "in-sample";
        training_row.standard_error_scope = "in_sample_approximate";
        training_row.evaluation_seed = base.training_seed;
        training_row.paths = training_row.evaluation_paths = config.training_paths;
        training_row.antithetic_pairs = antithetic ? config.training_paths / 2 : 0;
        training_row.unique_simulated_paths = training_row.payoff_evaluations = config.training_paths;
        training_row.time_zero_continuation = base.time_zero_continuation;
        add_regression_summary(training.policy, training_row);
        training_row.early_exercise_premium_bs = training_row.estimate.price - black_scholes(config.contract);
        rows.push_back(training_row);
    }
    if (out_of_sample) {
        Row row = base;
        row.valuation_mode = "out-of-sample";
        row.standard_error_scope = "conditional_on_fitted_policy";
        row.evaluation_seed = stream_seed(config.seed, 2);
        row.paths = row.evaluation_paths = config.paths;
        row.antithetic_pairs = antithetic ? config.paths / 2 : 0;
        row.unique_simulated_paths = row.payoff_evaluations = config.training_paths + config.paths;
        row.estimate = timed([&] {
            const Paths paths = simulate_paths(config.contract, config.paths, config.steps, row.evaluation_seed, sampling);
            const auto valued = evaluate_policy(config, training.policy, paths);
            return summarize_cashflows(config, paths, valued, sampling, row);
        }, row.valuation_runtime_ms);
        row.runtime_ms = row.training_runtime_ms + row.valuation_runtime_ms;
        row.early_exercise_premium_bs = row.estimate.price - black_scholes(config.contract);
        rows.push_back(row);
    }
    return rows;
}

int main(int argc, char **argv) {
    return run_experiment(argc, argv,
        "American option: common normalized LSM; plain/antithetic sampling and in/out-of-sample valuation.",
        lsm_experiment);
}
