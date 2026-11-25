// The fixtures call production functions directly; no second pricing algorithm.
#define main lsm_experiment_main
#include "lsm_american.cpp"
#undef main

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, double tolerance = 1e-12) {
    require(std::isfinite(actual) && std::isfinite(expected) && std::abs(actual - expected) <= tolerance,
            "Unexpected result: " + std::to_string(actual) + " expected " + std::to_string(expected));
}

Config example_config() {
    Config config;
    config.contract.spot = 1.0; config.contract.strike = 1.1;
    config.contract.rate = 0.06; config.contract.maturity = 3.0;
    config.steps = 3; config.paths = 8; config.training_paths = 8;
    return config;
}

Paths example_paths() {
    return {{1, 1, 1, 1, 1, 1, 1, 1},
            {1.09, 1.16, 1.22, 0.93, 1.11, 0.76, 0.92, 0.88},
            {1.08, 1.26, 1.07, 0.97, 1.56, 0.77, 0.84, 1.22},
            {1.34, 1.54, 1.03, 0.92, 1.52, 0.90, 1.01, 1.34}};
}

void published_eight_paths() {
    const auto config = example_config();
    const auto paths = example_paths();
    const auto trained = train_policy(config, paths, Basis::Quadratic);
    const std::vector<std::size_t> expected{3, 3, 3, 1, 3, 1, 1, 1};
    require(trained.valuation.stopping_times == expected, "Published stopping decisions differ.");
    const auto valued = evaluate_policy(config, trained.policy, paths);
    double price = 0.0;
    for (std::size_t path = 0; path < 8; ++path) {
        const double expected_pv = std::exp(-0.06 * static_cast<double>(expected[path])) *
                                  config.contract.payoff(paths[expected[path]][path]);
        near(trained.valuation.discounted_cashflows[path], expected_pv);
        near(valued.discounted_cashflows[path], expected_pv);
        price += expected_pv / 8.0;
    }
    near(price, (0.07 * std::exp(-0.18) + 0.91 * std::exp(-0.06)) / 8.0);
    near(price, 0.1144, 0.00005);
}

void antithetic_standard_errors() {
    Config config;
    config.contract.strike = 40; config.contract.rate = 0; config.paths = 4; config.steps = 1;
    const Paths paths{{40, 40, 40, 40}, {39, 37, 38, 34}};
    Cashflows cashflows{{2, 1, 4, 3}, {1, 1, 1, 1}};
    Row row;
    const auto av = summarize_cashflows(config, paths, cashflows, PathSampling::Antithetic, row);
    near(av.price, 2.5); near(av.standard_error, 1.0);
    require(av.samples == 2, "Wrong pair count.");
    near(row.naive_path_standard_error, std::sqrt(5.0 / 12.0));
    near(row.european_price, 3.0); near(row.european_standard_error, 1.0);
    near(row.european_naive_path_standard_error, std::sqrt(14.0 / 12.0));
    near(row.early_exercise_premium, -0.5); near(row.early_exercise_premium_standard_error, 0.0);
    near(row.early_exercise_premium_naive_path_standard_error, std::sqrt(17.0 / 12.0));
    const auto plain = summarize_cashflows(config, paths, cashflows, PathSampling::Plain, row);
    near(plain.standard_error, std::sqrt(5.0 / 12.0));
    require(plain.samples == 4, "Wrong independent path count.");
}

void time_zero() {
    Config config;
    config.contract.spot = 20; config.contract.strike = 40; config.contract.rate = 0.06;
    config.steps = 2;
    const Paths paths(3, std::vector<double>(4, 20));
    const auto trained = train_policy(config, paths);
    require(trained.policy.exercise_at_zero, "Immediate exercise must be available.");
    for (std::size_t path = 0; path < 4; ++path) {
        require(trained.valuation.stopping_times[path] == 0, "Wrong t=0 stop.");
        near(trained.valuation.discounted_cashflows[path], 20.0);
    }
}

void paths_and_sampling() {
    Contract c; c.rate = 0.06; c.volatility = 0.4; c.maturity = 2;
    const auto av = simulate_paths(c, 8, 7, 123, PathSampling::Antithetic);
    require(av == simulate_paths(c, 8, 7, 123, PathSampling::Antithetic), "AV is not reproducible.");
    const auto plain = simulate_paths(c, 8, 7, 123);
    require(plain != av, "Plain and AV paths should differ.");
    std::mt19937_64 generator(123); std::normal_distribution<double> normal;
    const double dt = c.maturity / 7.0;
    for (std::size_t time = 1; time <= 7; ++time) {
        for (std::size_t path = 0; path < 8; ++path) {
            const double expected = plain[time - 1][path] * std::exp(
                (c.rate - 0.5 * c.volatility * c.volatility) * dt + c.volatility * std::sqrt(dt) * normal(generator));
            near(plain[time][path], expected);
        }
        for (std::size_t pair = 0; pair < 4; ++pair) {
            const double log_sum = std::log(av[time][2 * pair] / c.spot) +
                                   std::log(av[time][2 * pair + 1] / c.spot);
            near(log_sum, 2 * (c.rate - 0.5 * c.volatility * c.volatility) * dt * static_cast<double>(time));
        }
    }
    bool rejected = false;
    try { simulate_paths(c, 7, 7, 123, PathSampling::Antithetic); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "Odd AV path count must fail.");
}

void scale_homogeneity() {
    for (const auto sampling : {PathSampling::Plain, PathSampling::Antithetic}) {
        Config a; a.contract.spot = 40; a.contract.strike = 40; a.steps = 5;
        Config b = a; b.contract.spot *= 2; b.contract.strike *= 2;
        const auto pa = simulate_paths(a.contract, 80, a.steps, 42, sampling);
        const auto pb = simulate_paths(b.contract, 80, b.steps, 42, sampling);
        const auto ta = train_policy(a, pa); const auto tb = train_policy(b, pb);
        require(ta.valuation.stopping_times == tb.valuation.stopping_times, "Scaling changed stopping decisions.");
        for (std::size_t time = 1; time < a.steps; ++time)
            for (std::size_t j = 0; j < 4; ++j) near(ta.policy.models[time].beta[j], tb.policy.models[time].beta[j]);
        const auto reused = evaluate_policy(b, ta.policy, pb);
        for (std::size_t i = 0; i < 80; ++i) {
            near(tb.valuation.discounted_cashflows[i], 2 * ta.valuation.discounted_cashflows[i]);
            near(reused.discounted_cashflows[i], tb.valuation.discounted_cashflows[i]);
        }
    }
}

void coefficient_reuse() {
    Config config; config.contract.spot = 40; config.contract.strike = 40; config.steps = 2;
    Policy policy; policy.models.resize(3); policy.models[1].valid = true; policy.models[1].beta[0] = 0.25;
    const Paths paths{{40, 40, 40, 40}, {32, 32, 32, 32}, {40, 40, 40, 40}};
    auto valued = evaluate_policy(config, policy, paths);
    for (auto time : valued.stopping_times) require(time == 2, "Compared dollars with normalized continuation.");
    policy.models[1].beta[0] = 0.10;
    valued = evaluate_policy(config, policy, paths);
    for (auto time : valued.stopping_times) require(time == 1, "Missed normalized exercise decision.");
    near(valued.discounted_cashflows[0], 8 * std::exp(-0.05 / 2));
}

void backward_updates() {
    Config config; config.contract.spot = 40; config.contract.strike = 40;
    config.contract.rate = 0.1; config.contract.maturity = 3; config.steps = 3;
    Paths paths{{40,40,40,40}, {36,36,36,36}, {30,30,30,30}, {39,39,39,39}};
    const auto trained = train_policy(config, paths);
    near(trained.policy.models[1].beta[0], 0.25 * std::exp(-0.1));
    for (std::size_t path = 0; path < 4; ++path) {
        require(trained.valuation.stopping_times[path] == 2, "Lost selected future stopping time.");
        near(trained.valuation.discounted_cashflows[path], 10 * std::exp(-0.2));
    }
}

void degeneracy() {
    const auto empty = fit_continuation({}, {});
    require(!empty.valid && empty.rank == 0 && empty.status == "no_itm", "Empty regression status.");
    const auto small = fit_continuation({0.8, 0.9}, {0.1, 0.3});
    require(small.rank == 1 && small.status == "constant_fallback", "Small ITM fallback missing.");
    near(small.continuation(0.85), 0.2);
    const auto deficient = fit_continuation(std::vector<double>(8, 0.8), std::vector<double>(8, 0.3));
    require(deficient.rank == 1 && deficient.status == "rank_deficient", "Rank deficiency not reported.");
    near(deficient.continuation(0.8), 0.3); near(deficient.condition_inf, 1.0);
}

void qr_diagnostics() {
    const std::vector<double> x{0.4,0.5,0.6,0.7,0.8,0.9};
    const std::array<double,4> beta{0.1,0.2,-0.03,0.02};
    std::vector<double> y;
    for (auto state : x) {
        const auto values = basis_values(state, 0, 1, Basis::Laguerre);
        y.push_back(std::inner_product(beta.begin(), beta.end(), values.begin(), 0.0));
    }
    const auto model = fit_continuation(x, y);
    require(model.rank == 4 && model.itm_samples == 6 && model.status == "full_rank", "Unexpected QR rank.");
    require(std::isfinite(model.condition_inf) && model.condition_inf > 1, "Missing conditioning diagnostic.");
    for (std::size_t j = 0; j < 4; ++j) near(model.beta[j], beta[j], 1e-8);
    for (std::size_t i = 0; i < x.size(); ++i) near(model.continuation(x[i]), y[i]);
    // The paper basis becomes ill-conditioned when the ITM stock range is narrow.
    // Test prediction accuracy there, rather than demanding stable individual betas.
    std::vector<double> narrow_x, narrow_y;
    for (std::size_t i = 0; i < 200; ++i) {
        const double state = 0.97 + 0.03 * static_cast<double>(i) / 199.0;
        const auto values = basis_values(state, 0, 1, Basis::Laguerre);
        narrow_x.push_back(state);
        narrow_y.push_back(std::inner_product(beta.begin(), beta.end(), values.begin(), 0.0));
    }
    const auto narrow = fit_continuation(narrow_x, narrow_y);
    require(narrow.rank == 4 && narrow.condition_inf > 1e8, "Narrow-range conditioning fixture.");
    for (std::size_t i = 0; i < narrow_x.size(); ++i) near(narrow.continuation(narrow_x[i]), narrow_y[i], 5e-12);
}

void frozen_policy() {
    Config config; config.steps = 5;
    for (auto sampling : {PathSampling::Plain, PathSampling::Antithetic}) {
        const auto paths = simulate_paths(config.contract, 100, config.steps, 42, sampling);
        const auto trained = train_policy(config, paths);
        const auto valued = evaluate_policy(config, trained.policy, paths);
        require(valued.stopping_times == trained.valuation.stopping_times, "Frozen stopping times differ.");
        for (std::size_t i = 0; i < 100; ++i)
            near(valued.discounted_cashflows[i], trained.valuation.discounted_cashflows[i]);
    }
}

void metadata() {
    Config config; config.steps = 4; config.reference_steps = 200; config.paths = 1000;
    config.training_paths = 1200; config.valuation = "both";
    for (const auto &sampling : {std::string("plain"), std::string("antithetic")}) {
        config.sampling = sampling;
        const auto rows = lsm_experiment(config);
        require(rows.size() == 2, "Expected both valuation modes.");
        require(rows[0].paths == 1200 && rows[1].paths == 1000, "Evaluation path accounting.");
        require(rows[0].unique_simulated_paths == 1200 && rows[1].unique_simulated_paths == 2200, "Unique path accounting.");
        require(rows[0].standard_error_scope == "in_sample_approximate", "In-sample SE caveat.");
        require(rows[1].standard_error_scope == "conditional_on_fitted_policy", "Out-of-sample SE scope.");
        require(rows[0].training_seed == rows[0].evaluation_seed, "In-sample stream.");
        require(rows[1].training_seed != rows[1].evaluation_seed, "Evaluation stream reused.");
        require(rows[1].sampling_method == sampling && rows[1].basis == "weighted_laguerre_4", "Method metadata.");
        require(rows[1].antithetic_pairs == (sampling == "antithetic" ? 500u : 0u), "Pair count.");
        require(rows[0].regression_rank_max > 0 && rows[0].regression_condition_inf_max > 0, "QR summary missing.");
    }
}

void exercise_grid() {
    Config config; config.contract.spot = 36; config.contract.strike = 40; config.contract.volatility = 0;
    config.steps = 4; config.reference_steps = 200; config.paths = 100; config.training_paths = 100;
    config.valuation = "both";
    for (auto row : lsm_experiment(config)) {
        near(row.estimate.price, 4.0); near(row.benchmark, 4.0); near(row.estimate.standard_error, 0.0);
        require(row.exercise_at_zero, "CRR/LSM t=0 opportunity differs.");
    }
    config.reference_steps = 201;
    bool rejected = false;
    try { lsm_experiment(config); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "Unaligned reference grid accepted.");
}

void no_test_max() {
    Config config; config.contract.spot = 20; config.contract.strike = 40; config.steps = 2;
    Policy policy; policy.models.resize(3); policy.models[1].valid = true; policy.models[1].beta[0] = 1.0;
    const Paths paths{{20,20,20,20}, {30,30,30,30}, {40,40,40,40}};
    const auto valued = evaluate_policy(config, policy, paths);
    for (auto value : valued.discounted_cashflows) near(value, 0.0);
}

void reproducible() {
    Config config; config.sampling = "plain"; config.steps = 4; config.reference_steps = 200;
    config.paths = 1001; config.training_paths = 1201; config.valuation = "both";
    const auto a = lsm_experiment(config); const auto b = lsm_experiment(config);
    for (std::size_t i = 0; i < 2; ++i) {
        near(a[i].estimate.price, b[i].estimate.price, 0);
        near(a[i].estimate.standard_error, b[i].estimate.standard_error, 0);
        require(a[i].training_seed == b[i].training_seed && a[i].evaluation_seed == b[i].evaluation_seed, "Seed reproducibility.");
    }
}

void deterministic() {
    Config config; config.steps = 4; config.reference_steps = 200; config.paths = 100;
    config.training_paths = 100; config.valuation = "both";
    config.contract.maturity = 0; config.contract.spot = 90;
    for (const auto &row : lsm_experiment(config)) {
        near(row.estimate.price, 10); near(row.estimate.standard_error, 0); near(row.exercise_fraction, 0);
    }
    config.contract.type = OptionType::Call; config.contract.maturity = 1;
    config.contract.volatility = 0; config.contract.spot = 100;
    for (const auto &row : lsm_experiment(config)) {
        near(row.estimate.price, black_scholes(config.contract)); near(row.estimate.standard_error, 0);
        require(!row.exercise_at_zero, "Nondividend call should be held.");
    }
}

void export_example(const std::filesystem::path &directory) {
    std::filesystem::create_directories(directory);
    auto config = example_config();
    const auto paths = example_paths();
    std::ofstream trace(directory / "example_decisions.csv"); write_trace_header(trace);
    const auto trained = train_policy(config, paths, Basis::Quadratic, &trace);
    std::ofstream diagnostics(directory / "example_regressions.csv"); write_regressions(diagnostics, config, trained.policy);
    std::ofstream coefficients(directory / "example_coefficients_monetary.csv");
    coefficients << "time_index,constant,stock,stock_squared\n" << std::setprecision(17);
    for (std::size_t time = 1; time < config.steps; ++time) {
        const auto &m = trained.policy.models[time];
        const double a2 = m.beta[2] / (config.contract.strike * m.scale * m.scale);
        const double a1 = m.beta[1] / m.scale - 2 * m.beta[2] * m.center / (m.scale * m.scale);
        const double a0 = config.contract.strike * (m.beta[0] - m.beta[1] * m.center / m.scale +
                          m.beta[2] * m.center * m.center / (m.scale * m.scale));
        coefficients << time << ',' << a0 << ',' << a1 << ',' << a2 << '\n';
    }
    std::ofstream cashflows(directory / "example_cashflows.csv");
    cashflows << "path,stopping_time,payoff,discounted_payoff\n" << std::setprecision(17);
    for (std::size_t path = 0; path < 8; ++path) {
        const auto time = trained.valuation.stopping_times[path];
        cashflows << path + 1 << ',' << time << ',' << config.contract.payoff(paths[time][path])
                  << ',' << trained.valuation.discounted_cashflows[path] << '\n';
    }
    const double price = std::accumulate(trained.valuation.discounted_cashflows.begin(),
        trained.valuation.discounted_cashflows.end(), 0.0) / 8.0;
    std::cout << "Published eight-path example: " << std::setprecision(12) << price << '\n';
}

int main(int argc, char **argv) {
    try {
        const std::string test = argc > 1 ? argv[1] : "";
        if (test == "published") published_eight_paths();
        else if (test == "standard_errors") antithetic_standard_errors();
        else if (test == "time_zero") time_zero();
        else if (test == "paths") paths_and_sampling();
        else if (test == "scale") scale_homogeneity();
        else if (test == "reuse") coefficient_reuse();
        else if (test == "backward") backward_updates();
        else if (test == "degeneracy") degeneracy();
        else if (test == "qr") qr_diagnostics();
        else if (test == "frozen") frozen_policy();
        else if (test == "metadata") metadata();
        else if (test == "grid") exercise_grid();
        else if (test == "no_test_max") no_test_max();
        else if (test == "reproducible") reproducible();
        else if (test == "deterministic") deterministic();
        else if (test == "example" && argc == 3) export_example(argv[2]);
        else throw std::runtime_error("Unknown procedure fixture.");
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
