"""Check LSM decisions against the paper's fixed paths and exact SE arithmetic."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LsmProcedureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            raise unittest.SkipTest("Procedure fixture requires g++ or clang++.")
        cls.directory = tempfile.TemporaryDirectory(prefix="lsm-procedure-")
        directory = Path(cls.directory.name)
        source = directory / "lsm_procedure.cpp"
        source.write_text(r'''
#define main lsm_experiment_main
#include "lsm_american.cpp"
#undef main

#include <iostream>

void near(double actual, double expected, double tolerance = 1e-12) {
    if (std::abs(actual - expected) > tolerance)
        throw std::runtime_error("Unexpected result: " + std::to_string(actual) +
                                 " expected " + std::to_string(expected));
}

void published_eight_paths() {
    // Longstaff and Schwartz (2001), Section 1.1, pp. 117-119.
    Config config;
    config.contract.spot = 1.0;
    config.contract.strike = 1.1;
    config.contract.rate = 0.06;
    config.contract.maturity = 3.0;
    config.steps = 3;
    config.paths = 8;
    const Paths paths{
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1.09, 1.16, 1.22, 0.93, 1.11, 0.76, 0.92, 0.88},
        {1.08, 1.26, 1.07, 0.97, 1.56, 0.77, 0.84, 1.22},
        {1.34, 1.54, 1.03, 0.92, 1.52, 0.90, 1.01, 1.34},
    };
    // The worked example uses ordinary 1, X, X^2; Table 1 uses Laguerre.
    const auto trained = train_policy(config, paths, false);
    const std::array<std::size_t, 8> expected{3, 3, 3, 1, 3, 1, 1, 1};
    double present_value = 0.0;
    for (std::size_t path = 0; path < 8; ++path) {
        std::size_t stopping_time = config.steps;
        for (std::size_t time = 1; time < config.steps; ++time) {
            const double exercise = config.contract.payoff(paths[time][path]);
            if (exercise > 0.0 && exercise > trained.policy.models[time].continuation(
                    paths[time][path] / config.contract.strike)) {
                stopping_time = time;
                break;
            }
        }
        if (stopping_time != expected[path])
            throw std::runtime_error("Published stopping rule not reproduced.");
        present_value += std::exp(-0.06 * static_cast<double>(stopping_time)) *
                         config.contract.payoff(paths[stopping_time][path]) / 8.0;
    }
    near(present_value, (0.07 * std::exp(-0.18) + 0.91 * std::exp(-0.06)) / 8.0);
    near(present_value, 0.1144, 0.00005);
}

void antithetic_standard_errors() {
    Config config;
    config.contract.strike = 40;
    config.contract.rate = 0;
    config.paths = 4;
    config.steps = 1;
    const Paths paths{{40, 40, 40, 40}, {39, 37, 38, 34}};
    TrainingResult trained;
    trained.discounted_cashflows = {2, 1, 4, 3};
    trained.stopping_times = {1, 1, 1, 1};
    Row row;
    const Estimate estimate = evaluate_in_sample_antithetic(config, paths, trained, row);
    near(estimate.price, 2.5);
    near(estimate.standard_error, 1.0); // Pair means 1.5, 3.5.
    if (estimate.samples != 2) throw std::runtime_error("Wrong pair count.");
    near(row.naive_path_standard_error, std::sqrt(5.0 / 12.0));
    near(row.european_price, 3.0);
    near(row.european_standard_error, 1.0); // Pair means 2, 4.
    near(row.european_naive_path_standard_error, std::sqrt(14.0 / 12.0));
    near(row.early_exercise_premium, -0.5);
    near(row.early_exercise_premium_standard_error, 0.0); // Both pair differences -0.5.
    near(row.early_exercise_premium_naive_path_standard_error, std::sqrt(17.0 / 12.0));
}

void normalized_cashflows_and_paper_grid() {
    Config config;
    config.contract.spot = 20;
    config.contract.strike = 40;
    config.contract.rate = 0.06;
    config.contract.maturity = 1;
    config.steps = 2;
    config.paths = 4;
    const Paths paths(3, std::vector<double>(4, 20));
    const auto trained = train_policy(config, paths, true);
    if (trained.policy.exercise_at_zero)
        throw std::runtime_error("Paper exercise grid starts after zero.");
    for (std::size_t path = 0; path < 4; ++path) {
        if (trained.stopping_times[path] != 1)
            throw std::runtime_error("Wrong first exercise date.");
        near(trained.discounted_cashflows[path], 20 * std::exp(-0.03));
    }
}

int main(int argc, char **argv) {
    try {
        const std::string test = argc > 1 ? argv[1] : "";
        if (test == "published") published_eight_paths();
        else if (test == "standard_errors") antithetic_standard_errors();
        else if (test == "normalization") normalized_cashflows_and_paper_grid();
        else throw std::runtime_error("Unknown procedure fixture.");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
''', encoding="utf-8")
        cls.binary = directory / "lsm_procedure.exe"
        subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic",
                        "-I", str(ROOT / "src"), "-I", str(ROOT / "include"),
                        str(source), "-o", str(cls.binary)], check=True, capture_output=True,
                       text=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def fixture(self, name):
        result = subprocess.run([str(self.binary), name], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_original_eight_path_stopping_rule(self):
        self.fixture("published")

    def test_pair_and_naive_standard_error_accounting(self):
        self.fixture("standard_errors")

    def test_cashflow_normalization_and_published_exercise_grid(self):
        self.fixture("normalization")


if __name__ == "__main__":
    unittest.main()
