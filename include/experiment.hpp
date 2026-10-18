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
    std::uint64_t seed = 42;
    double boundary_ratio = 0.85;
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
        else if (name == "--seed") config.seed = integer(value, name);
        else if (name == "--boundary-ratio") config.boundary_ratio = real(value, name);
        else if (name == "--output") config.output = value;
        else if (name == "--newton-trace") config.newton_trace = value;
        else if (name == "--type") {
            if (value != "call" && value != "put")
                throw std::invalid_argument("--type must be call or put.");
            config.contract.type = value == "call" ? OptionType::Call : OptionType::Put;
        } else throw std::invalid_argument("Unknown argument: " + name);
    }
    config.contract.validate();
    if (config.steps == 0 || config.steps > 20000 || config.reference_steps == 0 ||
        config.reference_steps > 20000)
        throw std::invalid_argument("Steps/reference steps must be in [1, 20000].");
    if (config.paths < 4 || config.paths > 10000000 || config.training_paths < 4 ||
        config.training_paths > 5000000)
        throw std::invalid_argument("Paths must be in [4, 10000000]; training paths in [4, 5000000].");
    if (config.boundary_ratio <= 0.0)
        throw std::invalid_argument("Boundary ratio must be positive.");
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
    double newton_log_second_moment_initial = 0.0;
    double newton_log_second_moment_final = 0.0;
    std::string newton_status = "not_applicable";
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
           "newton_log_second_moment_initial,newton_log_second_moment_final,newton_status\n";
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
            << row.newton_log_second_moment_initial << ',' << row.newton_log_second_moment_final
            << ',' << row.newton_status << '\n';
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
                "--boundary-ratio X    fixed-rule MC: put <= X*K, call >= X*K; default 0.85\n"
                "--seed N              master random seed; default 42\n"
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
