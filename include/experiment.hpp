#pragma once

#include "binomial.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

namespace pricing {

struct Config {
    Contract contract;
    std::size_t steps = 50;
    std::size_t paths = 100000;
    std::size_t training_paths = 50000;
    std::size_t reference_steps = 2000;
    std::size_t exercise_dates = 0; // Optional matched-date CRR output; zero means every tree date only.
    std::uint64_t seed = 42;
    double boundary_ratio = 0.85;
    std::string lsm_mode; // Legacy aliases, resolved only by the LSM executable.
    std::string sampling = "antithetic";
    std::string valuation = "in-sample";
    bool sampling_explicit = false;
    bool valuation_explicit = false;
    std::string regression_diagnostics;
    std::string lsm_trace;
    std::string output;
    std::string newton_trace;
};

inline std::size_t integer(const std::string &value, const std::string &name) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument(name + " requires a nonnegative integer.");
    std::size_t end = 0;
    const auto result = std::stoull(value, &end);
    if (end != value.size() || result > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Invalid integer for " + name);
    return static_cast<std::size_t>(result);
}

inline double real(const std::string &value, const std::string &name) {
    std::size_t end = 0;
    const double result = std::stod(value, &end);
    if (end != value.size() || !std::isfinite(result))
        throw std::invalid_argument(name + " requires a finite number.");
    return result;
}

inline Config parse_config(int argc, char **argv) {
    Config config;
    for (int i = 1; i < argc; i += 2) {
        const std::string name = argv[i];
        if (i + 1 == argc)
            throw std::invalid_argument("Missing value for " + name);
        const std::string value = argv[i + 1];
        if (name == "--spot") config.contract.spot = real(value, name);
        else if (name == "--strike") config.contract.strike = real(value, name);
        else if (name == "--rate") config.contract.rate = real(value, name);
        else if (name == "--volatility") config.contract.volatility = real(value, name);
        else if (name == "--maturity") config.contract.maturity = real(value, name);
        else if (name == "--steps") config.steps = integer(value, name);
        else if (name == "--paths") config.paths = integer(value, name);
        else if (name == "--train-paths") config.training_paths = integer(value, name);
        else if (name == "--reference-steps") config.reference_steps = integer(value, name);
        else if (name == "--exercise-dates") config.exercise_dates = integer(value, name);
        else if (name == "--seed") config.seed = integer(value, name);
        else if (name == "--boundary-ratio") config.boundary_ratio = real(value, name);
        else if (name == "--lsm-mode") config.lsm_mode = value;
        else if (name == "--sampling") { config.sampling = value; config.sampling_explicit = true; }
        else if (name == "--valuation") { config.valuation = value; config.valuation_explicit = true; }
        else if (name == "--regression-diagnostics") config.regression_diagnostics = value;
        else if (name == "--lsm-trace") config.lsm_trace = value;
        else if (name == "--output") config.output = value;
        else if (name == "--newton-trace") config.newton_trace = value;
        else if (name == "--type") {
            if (value != "call" && value != "put")
                throw std::invalid_argument("--type must be call or put.");
            config.contract.type = value == "call" ? OptionType::Call : OptionType::Put;
        } else throw std::invalid_argument("Unknown argument: " + name);
    }
    config.contract.validate();
    if (config.steps == 0 || config.steps > 100000 || config.reference_steps == 0 ||
        config.reference_steps > 100000)
        throw std::invalid_argument("Steps/reference steps must be in [1, 100000].");
    if (config.paths < 4 || config.paths > 10000000 || config.training_paths < 4 ||
        config.training_paths > 5000000)
        throw std::invalid_argument("Paths must be in [4, 10000000]; training paths in [4, 5000000].");
    if (config.boundary_ratio <= 0.0)
        throw std::invalid_argument("Boundary ratio must be positive.");
    if (!config.lsm_mode.empty() && config.lsm_mode != "independent" && config.lsm_mode != "paper")
        throw std::invalid_argument("--lsm-mode must be independent or paper.");
    return config;
}

struct Row {
    std::string method;
    Estimate estimate;
    double benchmark = 0.0;
    std::size_t steps = 0;
    std::size_t paths = 0;
    std::size_t training_paths = 0;
    std::size_t payoff_evaluations = 0;
    double runtime_ms = 0.0;
    double control_beta = 0.0;
    double importance_shift = 0.0;
    double exercise_fraction = 0.0;
    std::size_t reference_steps = 0;
    std::size_t newton_iterations = 0;
    double newton_gradient = 0.0;
    double newton_second_moment_initial = 0.0;
    double newton_second_moment_final = 0.0;
    std::string newton_status = "not_applicable";
    // Supplemental diagnostics used by the original-paper LSM benchmark.
    double naive_path_standard_error = 0.0;
    double european_price = 0.0;
    double european_standard_error = 0.0;
    double european_naive_path_standard_error = 0.0;
    double early_exercise_premium = 0.0;
    double early_exercise_premium_standard_error = 0.0;
    double early_exercise_premium_naive_path_standard_error = 0.0;
    double early_exercise_premium_bs = 0.0;
    double american_reference_crr = 0.0;
    double reference_runtime_ms = 0.0;
    std::string sampling_method = "not_applicable";
    std::string valuation_mode = "not_applicable";
    std::string basis = "not_applicable";
    std::string standard_error_scope = "not_applicable";
    std::uint64_t training_seed = 0;
    std::uint64_t evaluation_seed = 0;
    std::size_t evaluation_paths = 0;
    std::size_t antithetic_pairs = 0;
    std::size_t training_antithetic_pairs = 0;
    std::size_t unique_simulated_paths = 0;
    bool exercise_at_zero = false;
    double time_zero_continuation = 0.0;
    std::size_t regression_rank_min = 0;
    std::size_t regression_rank_max = 0;
    std::size_t regression_itm_min = 0;
    std::size_t regression_itm_max = 0;
    std::size_t regression_fallback_dates = 0;
    std::size_t regression_no_itm_dates = 0;
    double regression_condition_inf_max = 0.0;
    double training_runtime_ms = 0.0;
    double valuation_runtime_ms = 0.0;
};

template <class Function> inline auto timed(Function function, double &milliseconds) {
    const auto start = std::chrono::steady_clock::now();
    auto result = function();
    milliseconds = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - start).count();
    return result;
}

inline void write_csv(std::ostream &out, const Config &config, const std::vector<Row> &rows) {
    out << "method,option_type,spot,strike,rate,volatility,maturity,steps,paths,training_paths,"
           "independent_samples,payoff_evaluations,seed,price,standard_error,ci_lower,ci_upper,"
           "benchmark,absolute_error,runtime_ms,control_beta,importance_shift,boundary_ratio,"
           "exercise_fraction,reference_steps,newton_iterations,newton_gradient,"
           "newton_second_moment_initial,newton_second_moment_final,newton_status,"
           "naive_path_standard_error,european_price,european_standard_error,"
           "european_naive_path_standard_error,early_exercise_premium,"
           "early_exercise_premium_standard_error,early_exercise_premium_naive_path_standard_error,"
           "early_exercise_premium_bs,american_reference_crr,reference_runtime_ms,"
           "sampling_method,valuation_mode,basis,standard_error_scope,training_seed,evaluation_seed,"
           "evaluation_paths,antithetic_pairs,training_antithetic_pairs,unique_simulated_paths,"
           "exercise_at_zero,time_zero_continuation,regression_rank_min,regression_rank_max,"
           "regression_itm_min,regression_itm_max,regression_fallback_dates,regression_no_itm_dates,"
           "regression_condition_inf_max,training_runtime_ms,valuation_runtime_ms\n";
    out << std::setprecision(17);
    const auto &c = config.contract;
    constexpr double z95 = 1.959963984540054;
    for (const auto &row : rows) {
        const auto &e = row.estimate;
        if (!std::isfinite(e.price) || !std::isfinite(e.standard_error) ||
            !std::isfinite(row.benchmark))
            throw std::overflow_error("Non-finite experiment output.");
        out << row.method << ',' << (c.type == OptionType::Call ? "call" : "put") << ','
            << c.spot << ',' << c.strike << ',' << c.rate << ',' << c.volatility << ','
            << c.maturity << ',' << row.steps << ',' << row.paths << ',' << row.training_paths
            << ',' << e.samples << ',' << row.payoff_evaluations << ',' << config.seed << ','
            << e.price << ',' << e.standard_error << ',' << e.price - z95 * e.standard_error
            << ',' << e.price + z95 * e.standard_error << ',' << row.benchmark << ','
            << std::abs(e.price - row.benchmark) << ',' << row.runtime_ms << ','
            << row.control_beta << ',' << row.importance_shift << ',' << config.boundary_ratio
            << ',' << row.exercise_fraction << ',' << row.reference_steps << ','
            << row.newton_iterations << ',' << row.newton_gradient << ','
            << row.newton_second_moment_initial << ',' << row.newton_second_moment_final
            << ',' << row.newton_status << ',' << row.naive_path_standard_error
            << ',' << row.european_price << ',' << row.european_standard_error
            << ',' << row.european_naive_path_standard_error << ',' << row.early_exercise_premium
            << ',' << row.early_exercise_premium_standard_error
            << ',' << row.early_exercise_premium_naive_path_standard_error
            << ',' << row.early_exercise_premium_bs << ',' << row.american_reference_crr
            << ',' << row.reference_runtime_ms << ',' << row.sampling_method << ',' << row.valuation_mode
            << ',' << row.basis << ',' << row.standard_error_scope << ',' << row.training_seed
            << ',' << row.evaluation_seed << ',' << row.evaluation_paths << ',' << row.antithetic_pairs
            << ',' << row.training_antithetic_pairs << ',' << row.unique_simulated_paths
            << ',' << row.exercise_at_zero << ',' << row.time_zero_continuation
            << ',' << row.regression_rank_min << ',' << row.regression_rank_max
            << ',' << row.regression_itm_min << ',' << row.regression_itm_max
            << ',' << row.regression_fallback_dates << ',' << row.regression_no_itm_dates
            << ',' << row.regression_condition_inf_max << ',' << row.training_runtime_ms
            << ',' << row.valuation_runtime_ms << '\n';
    }
}

template <class Function>
inline int run_experiment(int argc, char **argv, const std::string &description, Function function) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << description << "\n\n"
                "All methods assume a non-dividend-paying stock (q=0).\n"
                "--type call|put        default put\n"
                "--spot X --strike X    default 100, 100\n"
                "--rate X              continuously compounded; default 0.05\n"
                "--volatility X        annualized; default 0.20\n"
                "--maturity X          years; default 1\n"
                "--steps N             tree steps / MC exercise intervals; default 50\n"
                "--paths N             evaluation paths; default 100000\n"
                "--train-paths N       LSM training / control and IS pilots; default 50000\n"
                "--reference-steps N   American CRR reference resolution; default 2000\n"
                "--exercise-dates N    binomial_american: also export a matched-date reference\n"
                "--boundary-ratio X    fixed-rule MC: put <= X*K, call >= X*K; default 0.85\n"
                "--seed N              master random seed; default 42\n"
                "--sampling MODE       LSM: plain or antithetic (default)\n"
                "--valuation MODE      LSM: in-sample (default), out-of-sample, or both\n"
                "--lsm-mode MODE       legacy LSM aliases: paper or independent\n"
                "--regression-diagnostics PATH  LSM per-date QR diagnostics CSV\n"
                "--lsm-trace PATH       LSM backward decisions CSV (small learning runs)\n"
                "--output PATH         CSV destination; default stdout\n"
                "--newton-trace PATH   variance_reduction only: Newton iteration CSV\n"
                "Method-specific use of these shared options is documented in README.md.\n";
            return 0;
        }
        const Config config = parse_config(argc, argv);
        const auto rows = function(config);
        if (config.output.empty()) write_csv(std::cout, config, rows);
        else {
            const std::filesystem::path destination(config.output);
            if (destination.has_parent_path())
                std::filesystem::create_directories(destination.parent_path());
            std::ofstream output(destination);
            if (!output) throw std::runtime_error("Cannot open CSV: " + config.output);
            write_csv(output, config, rows);
            output.flush();
            if (!output) throw std::runtime_error("Cannot write CSV: " + config.output);
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}

} // namespace pricing
