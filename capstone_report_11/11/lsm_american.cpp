#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Settings {
    double spot = 40.0;
    double strike = 40.0;
    double rate = 0.06;
    double volatility = 0.20;
    double maturity = 1.0;
    std::size_t steps = 50;
    std::size_t paths = 100000;
    std::size_t reference_steps = 2000;
    std::size_t boundary_points = 1000;
    std::size_t plotted_paths = 32;
    std::uint64_t seed = 42;
    std::string sampling = "antithetic";
    std::string output_dir = ".";
    bool reference_only = false;
    bool visualization_data = false;
};

struct Regression {
    std::array<double, 4> coefficients{};
    bool valid = false;

    double continuation(double stock) const {
        const double weight = std::exp(-0.5 * stock);
        const double fitted =
            coefficients[0] +
            weight * (coefficients[1] + coefficients[2] * (1.0 - stock) +
                      coefficients[3] * (1.0 - 2.0 * stock + 0.5 * stock * stock));
        return std::max(0.0, fitted);
    }
};

struct Moments {
    std::size_t count = 0;
    double mean = 0.0;
    double sum_squares = 0.0;

    void add(double value) {
        ++count;
        const double delta = value - mean;
        mean += delta / static_cast<double>(count);
        sum_squares += delta * (value - mean);
    }

    double standard_error() const {
        if (count < 2)
            throw std::runtime_error("At least two independent observations are required.");
        const double variance = sum_squares / static_cast<double>(count - 1);
        return std::sqrt(std::max(0.0, variance) / static_cast<double>(count));
    }
};

using Paths = std::vector<std::vector<double>>;

double payoff(double stock, double strike) { return std::max(strike - stock, 0.0); }

std::array<double, 4> basis(double stock) {
    const double weight = std::exp(-0.5 * stock);
    return {1.0, weight, weight * (1.0 - stock),
            weight * (1.0 - 2.0 * stock + 0.5 * stock * stock)};
}

Regression fit_continuation(const std::vector<double> &stocks,
                            const std::vector<double> &targets) {
    Regression model;
    constexpr std::size_t columns = 4;
    if (stocks.size() < columns)
        return model;
    model.valid = true;

    std::array<std::vector<double>, columns> q;
    std::array<double, columns> column_scales{};
    std::array<std::array<double, columns>, columns> r{};
    std::array<double, columns> q_transpose_y{};
    for (std::size_t column = 0; column < columns; ++column) {
        q[column].reserve(stocks.size());
        for (double stock : stocks)
            q[column].push_back(basis(stock)[column]);
        double scale_squared = 0.0;
        for (double value : q[column])
            scale_squared += value * value;
        const double scale = std::sqrt(scale_squared);
        if (scale == 0.0 || !std::isfinite(scale)) {
            model.valid = false;
            return model;
        }
        column_scales[column] = scale;
        for (double &value : q[column])
            value /= scale;

        for (int pass = 0; pass < 2; ++pass) {
            for (std::size_t previous = 0; previous < column; ++previous) {
                double projection = 0.0;
                for (std::size_t i = 0; i < stocks.size(); ++i)
                    projection += q[previous][i] * q[column][i];
                r[previous][column] += projection;
                for (std::size_t i = 0; i < stocks.size(); ++i)
                    q[column][i] -= projection * q[previous][i];
            }
        }

        double norm_squared = 0.0;
        for (double value : q[column])
            norm_squared += value * value;
        const double norm = std::sqrt(norm_squared);
        if (norm < 1e-10 * std::sqrt(static_cast<double>(stocks.size()))) {
            model.valid = false;
            return model;
        }
        r[column][column] = norm;
        for (double &value : q[column])
            value /= norm;
        for (std::size_t i = 0; i < stocks.size(); ++i)
            q_transpose_y[column] += q[column][i] * targets[i];
    }

    std::array<double, columns> scaled_coefficients{};
    for (std::size_t i = columns; i-- > 0;) {
        double rhs = q_transpose_y[i];
        for (std::size_t j = i + 1; j < columns; ++j)
            rhs -= r[i][j] * scaled_coefficients[j];
        scaled_coefficients[i] = rhs / r[i][i];
    }
    for (std::size_t i = 0; i < columns; ++i)
        model.coefficients[i] = scaled_coefficients[i] / column_scales[i];
    return model;
}

Paths simulate_paths(const Settings &settings) {
    if (settings.paths < 4)
        throw std::invalid_argument("At least four simulation paths are required.");
    if (settings.sampling == "antithetic" && settings.paths % 2 != 0)
        throw std::invalid_argument("Antithetic sampling requires an even path count.");
    if (settings.steps == 0 || settings.steps > 2000)
        throw std::invalid_argument("Exercise steps must be in [1, 2000].");
    if (settings.paths > 50000000 / (settings.steps + 1))
        throw std::invalid_argument("Path matrix exceeds the 50-million-value memory limit.");

    Paths paths(settings.steps + 1, std::vector<double>(settings.paths, settings.spot));
    std::mt19937_64 generator(settings.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    const double dt = settings.maturity / static_cast<double>(settings.steps);
    const double drift = (settings.rate - 0.5 * settings.volatility * settings.volatility) * dt;
    const double diffusion = settings.volatility * std::sqrt(dt);
    for (std::size_t time = 1; time <= settings.steps; ++time) {
        if (settings.sampling == "antithetic") {
            for (std::size_t pair = 0; pair < settings.paths / 2; ++pair) {
                const double shock = diffusion * normal(generator);
                const std::size_t first = 2 * pair;
                const std::size_t second = first + 1;
                paths[time][first] = paths[time - 1][first] * std::exp(drift + shock);
                paths[time][second] = paths[time - 1][second] * std::exp(drift - shock);
            }
        } else {
            for (std::size_t path = 0; path < settings.paths; ++path) {
                const double shock = diffusion * normal(generator);
                paths[time][path] = paths[time - 1][path] * std::exp(drift + shock);
            }
        }
        for (std::size_t path = 0; path < settings.paths; ++path) {
            if (!std::isfinite(paths[time][path]))
                throw std::overflow_error("Simulated stock price is not finite.");
        }
    }
    return paths;
}

std::vector<Regression> train_policy(const Settings &settings, const Paths &paths) {
    std::vector<Regression> policy(settings.steps + 1);
    const double discount =
        std::exp(-settings.rate * settings.maturity / static_cast<double>(settings.steps));
    std::vector<double> future_cashflow(settings.paths);
    for (std::size_t path = 0; path < settings.paths; ++path)
        future_cashflow[path] = payoff(paths.back()[path], settings.strike);

    for (std::size_t time = settings.steps; --time > 0;) {
        std::vector<double> in_the_money_stocks;
        std::vector<double> discounted_future_values;
        in_the_money_stocks.reserve(settings.paths);
        discounted_future_values.reserve(settings.paths);
        for (std::size_t path = 0; path < settings.paths; ++path) {
            future_cashflow[path] *= discount;
            if (payoff(paths[time][path], settings.strike) > 0.0) {
                in_the_money_stocks.push_back(paths[time][path]);
                discounted_future_values.push_back(future_cashflow[path]);
            }
        }

        policy[time] = fit_continuation(in_the_money_stocks, discounted_future_values);
        if (!policy[time].valid)
            continue;
        for (std::size_t path = 0; path < settings.paths; ++path) {
            const double immediate = payoff(paths[time][path], settings.strike);
            if (immediate > 0.0 &&
                immediate > policy[time].continuation(paths[time][path]))
                future_cashflow[path] = immediate;
        }
    }
    double estimated_value_at_zero = 0.0;
    for (double cashflow : future_cashflow)
        estimated_value_at_zero += discount * cashflow;
    estimated_value_at_zero /= static_cast<double>(settings.paths);
    policy[0].valid = true;
    policy[0].coefficients[0] = estimated_value_at_zero;
    return policy;
}

struct Valuation {
    std::vector<std::size_t> stopping_time;
    double estimate = 0.0;
    double standard_error = 0.0;
    double exercise_fraction = 0.0;
};

Valuation value_policy(const Settings &settings, const Paths &paths,
                       const std::vector<Regression> &policy) {
    Valuation result;
    result.stopping_time.resize(settings.paths, settings.steps);
    std::vector<double> discounted_payoffs(settings.paths);
    const double dt = settings.maturity / static_cast<double>(settings.steps);
    std::size_t early_exercises = 0;

    for (std::size_t path = 0; path < settings.paths; ++path) {
        std::size_t stop = settings.steps;
        const double immediate_at_zero = payoff(settings.spot, settings.strike);
        if (immediate_at_zero > policy[0].continuation(settings.spot))
            stop = 0;
        for (std::size_t time = 1; stop == settings.steps && time < settings.steps; ++time) {
            const double immediate = payoff(paths[time][path], settings.strike);
            if (immediate > 0.0 && policy[time].valid &&
                immediate > policy[time].continuation(paths[time][path])) {
                stop = time;
                break;
            }
        }
        result.stopping_time[path] = stop;
        if (stop < settings.steps)
            ++early_exercises;
        discounted_payoffs[path] = std::exp(-settings.rate * static_cast<double>(stop) * dt) *
                                   payoff(paths[stop][path], settings.strike);
    }

    Moments observations;
    if (settings.sampling == "antithetic") {
        for (std::size_t pair = 0; pair < settings.paths / 2; ++pair)
            observations.add(0.5 *
                             (discounted_payoffs[2 * pair] + discounted_payoffs[2 * pair + 1]));
    } else {
        for (double value : discounted_payoffs)
            observations.add(value);
    }
    result.estimate = observations.mean;
    result.standard_error = observations.standard_error();
    result.exercise_fraction =
        static_cast<double>(early_exercises) / static_cast<double>(settings.paths);
    return result;
}

double bermudan_crr(const Settings &settings) {
    const std::size_t steps = settings.reference_steps;
    if (steps % settings.steps != 0)
        throw std::invalid_argument("CRR steps must be divisible by exercise dates.");
    const double dt = settings.maturity / static_cast<double>(steps);
    const double log_up = settings.volatility * std::sqrt(dt);
    const double up = std::exp(log_up);
    const double down = 1.0 / up;
    const double probability = (std::exp(settings.rate * dt) - down) / (up - down);
    if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0)
        throw std::invalid_argument("Invalid CRR probability.");

    std::vector<double> values(steps + 1);
    double stock = settings.spot * std::exp(-static_cast<double>(steps) * log_up);
    const double ratio = up * up;
    for (std::size_t j = 0; j <= steps; ++j, stock *= ratio)
        values[j] = payoff(stock, settings.strike);

    const double discount = std::exp(-settings.rate * dt);
    const std::size_t exercise_every = steps / settings.steps;
    for (std::size_t i = steps; i-- > 0;) {
        stock = settings.spot * std::exp(-static_cast<double>(i) * log_up);
        const bool can_exercise = i % exercise_every == 0;
        for (std::size_t j = 0; j <= i; ++j, stock *= ratio) {
            const double continuation =
                discount * ((1.0 - probability) * values[j] + probability * values[j + 1]);
            values[j] = can_exercise ? std::max(payoff(stock, settings.strike), continuation)
                                     : continuation;
        }
    }
    return values[0];
}

std::vector<double> exercise_boundary(const Settings &settings,
                                      const std::vector<Regression> &policy) {
    std::vector<double> boundary(settings.steps + 1, std::numeric_limits<double>::quiet_NaN());
    const double low = 0.01 * settings.strike;
    const double high = settings.strike;
    for (std::size_t time = 1; time < settings.steps; ++time) {
        if (!policy[time].valid)
            continue;
        for (std::size_t point = 0; point <= settings.boundary_points; ++point) {
            const double stock = low + (high - low) * static_cast<double>(point) /
                                           static_cast<double>(settings.boundary_points);
            if (payoff(stock, settings.strike) > policy[time].continuation(stock))
                boundary[time] = stock;
        }
    }
    boundary[settings.steps] = settings.strike;
    return boundary;
}

double number_argument(const std::string &text, const std::string &name) {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value))
        throw std::invalid_argument("Invalid number for " + name + ".");
    return value;
}

std::size_t size_argument(const std::string &text, const std::string &name) {
    const double value = number_argument(text, name);
    if (value < 1.0 || value > 50000000.0 || std::floor(value) != value)
        throw std::invalid_argument("Invalid positive integer for " + name + ".");
    return static_cast<std::size_t>(value);
}

Settings parse_arguments(int argc, char **argv) {
    Settings settings;
    for (int i = 1; i < argc; ++i) {
        const std::string name = argv[i];
        if (name == "--reference-only") {
            settings.reference_only = true;
            continue;
        }
        if (name == "--visualization-data") {
            settings.visualization_data = true;
            continue;
        }
        if (name == "--help") {
            std::cout << "Native Longstaff-Schwartz American put\n"
                         "Options: --spot X --strike X --rate X --volatility X --maturity X\n"
                         "         --steps N --paths N --seed N --sampling plain|antithetic\n"
                         "         --reference-steps N --boundary-points N --output-dir DIR\n"
                         "         --reference-only --visualization-data\n";
            std::exit(0);
        }
        if (i + 1 >= argc)
            throw std::invalid_argument("Missing value for " + name + ".");
        const std::string value = argv[++i];
        if (name == "--spot")
            settings.spot = number_argument(value, name);
        else if (name == "--strike")
            settings.strike = number_argument(value, name);
        else if (name == "--rate")
            settings.rate = number_argument(value, name);
        else if (name == "--volatility")
            settings.volatility = number_argument(value, name);
        else if (name == "--maturity")
            settings.maturity = number_argument(value, name);
        else if (name == "--steps")
            settings.steps = size_argument(value, name);
        else if (name == "--paths")
            settings.paths = size_argument(value, name);
        else if (name == "--seed")
            settings.seed = static_cast<std::uint64_t>(size_argument(value, name));
        else if (name == "--reference-steps")
            settings.reference_steps = size_argument(value, name);
        else if (name == "--boundary-points")
            settings.boundary_points = size_argument(value, name);
        else if (name == "--sampling")
            settings.sampling = value;
        else if (name == "--output-dir")
            settings.output_dir = value;
        else
            throw std::invalid_argument("Unknown option " + name + ".");
    }
    if (settings.spot <= 0.0 || settings.strike <= 0.0 || settings.volatility <= 0.0 ||
        settings.maturity <= 0.0 || settings.rate < 0.0)
        throw std::invalid_argument(
            "Spot, strike, volatility, and maturity must be positive; rate nonnegative.");
    if (settings.sampling != "plain" && settings.sampling != "antithetic")
        throw std::invalid_argument("--sampling must be plain or antithetic.");
    if (settings.reference_steps > 10000 && !settings.reference_only)
        throw std::invalid_argument("Use --reference-only for CRR trees above 10,000 steps.");
    if (settings.reference_steps % settings.steps != 0)
        throw std::invalid_argument("CRR steps must be divisible by exercise dates.");
    if (settings.boundary_points < 100)
        throw std::invalid_argument("At least 100 boundary grid intervals are required.");
    return settings;
}

void write_outputs(const Settings &settings, const Valuation &value,
                   const std::vector<double> &boundary, double reference, const Paths *paths) {
    std::filesystem::create_directories(settings.output_dir);
    const auto file_path = [&settings](const std::string &name) {
        return std::filesystem::path(settings.output_dir) / name;
    };
    std::ofstream summary(file_path("summary.csv"));
    if (!summary)
        throw std::runtime_error("Could not write summary.csv.");
    summary << "spot,strike,rate,volatility,maturity,steps,paths,sampling,seed,"
               "reference_steps,estimate,standard_error,ci_lower,ci_upper,"
               "crr_reference,absolute_error,exercise_fraction,independent_observations\n"
            << std::setprecision(17) << settings.spot << ',' << settings.strike << ','
            << settings.rate << ',' << settings.volatility << ',' << settings.maturity << ','
            << settings.steps << ',' << settings.paths << ',' << settings.sampling << ','
            << settings.seed << ',' << settings.reference_steps << ',' << value.estimate << ','
            << value.standard_error << ','
            << value.estimate - 1.959963984540054 * value.standard_error << ','
            << value.estimate + 1.959963984540054 * value.standard_error << ',' << reference << ','
            << std::abs(value.estimate - reference) << ',' << value.exercise_fraction << ','
            << (settings.sampling == "antithetic" ? settings.paths / 2 : settings.paths) << '\n';

    std::ofstream boundary_file(file_path("exercise_boundary.csv"));
    if (!boundary_file)
        throw std::runtime_error("Could not write exercise_boundary.csv.");
    boundary_file << "sampling,strike,step,time_years,time_to_maturity,boundary_stock\n"
                  << std::setprecision(17);
    for (std::size_t time = 1; time <= settings.steps; ++time) {
        const double elapsed =
            settings.maturity * static_cast<double>(time) / static_cast<double>(settings.steps);
        boundary_file << settings.sampling << ',' << settings.strike << ',' << time << ','
                      << elapsed << ',' << settings.maturity - elapsed << ',' << boundary[time]
                      << '\n';
    }

    if (paths != nullptr) {
        std::ofstream path_file(file_path("sample_paths.csv"));
        if (!path_file)
            throw std::runtime_error("Could not write sample_paths.csv.");
        path_file << "path_id,time_years,stock_price,stopping_time_years,exercised_early\n"
                  << std::setprecision(17);
        const std::size_t count = std::min(settings.plotted_paths, settings.paths);
        for (std::size_t id = 0; id < count; ++id) {
            const std::size_t stop = value.stopping_time[id];
            const double stop_years =
                settings.maturity * static_cast<double>(stop) / static_cast<double>(settings.steps);
            for (std::size_t time = 0; time <= settings.steps; ++time) {
                const double elapsed = settings.maturity * static_cast<double>(time) /
                                       static_cast<double>(settings.steps);
                path_file << id << ',' << elapsed << ',' << (*paths)[time][id] << ',' << stop_years
                          << ',' << (stop < settings.steps ? 1 : 0) << '\n';
            }
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Settings settings = parse_arguments(argc, argv);
        if (settings.reference_only) {
            std::cout << std::setprecision(17) << bermudan_crr(settings) << '\n';
            return 0;
        }
        const Paths paths = simulate_paths(settings);
        const std::vector<Regression> policy = train_policy(settings, paths);
        const Valuation value = value_policy(settings, paths, policy);
        const std::vector<double> boundary = exercise_boundary(settings, policy);
        const double reference = bermudan_crr(settings);
        write_outputs(settings, value, boundary, reference,
                      settings.visualization_data ? &paths : nullptr);
        std::cout << std::setprecision(8) << "LSM " << settings.sampling << ": " << value.estimate
                  << " +/- " << 1.959963984540054 * value.standard_error
                  << " (nominal 95% path-sampling interval)\n"
                  << "Bermudan CRR (" << settings.reference_steps << " steps): " << reference
                  << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
