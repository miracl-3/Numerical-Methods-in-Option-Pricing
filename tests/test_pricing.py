"""Financial identities, numerical accuracy, statistical accounting, and CLI validation."""

import argparse
import csv
import math
from pathlib import Path
import subprocess
import sys
import unittest
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build import METHODS, executable  # noqa: E402
from run_experiments import run  # noqa: E402

BUILD_DIR = ROOT / "build"


class PricingTests(unittest.TestCase):
    def price(self, method, **parameters):
        parameters.setdefault("reference_steps", 1000)
        return run(method, BUILD_DIR, **parameters)

    def test_exactly_six_source_files(self):
        self.assertEqual({p.stem for p in (ROOT / "src").glob("*.cpp")}, set(METHODS))

    def test_black_scholes_known_values(self):
        for kind, expected in [("call", 10.450583572185565), ("put", 5.573526022256971)]:
            row = self.price("binomial_european", type=kind)[0]
            self.assertAlmostEqual(float(row["benchmark"]), expected, places=10)

    def test_tree_convergence(self):
        for kind in ["call", "put"]:
            coarse = self.price("binomial_european", type=kind, steps=20)[0]
            fine = self.price("binomial_european", type=kind, steps=1000)[0]
            self.assertLess(float(fine["absolute_error"]), 0.003)
            self.assertLess(float(fine["absolute_error"]), float(coarse["absolute_error"]))

    def test_put_call_parity_multiple_maturities(self):
        for maturity in [0.25, 1.0, 2.0]:
            call = self.price("binomial_european", type="call", maturity=maturity, steps=250)[0]
            put = self.price("binomial_european", type="put", maturity=maturity, steps=250)[0]
            expected = 100 - 100 * math.exp(-0.05 * maturity)
            self.assertAlmostEqual(float(call["price"]) - float(put["price"]), expected, places=9)
            self.assertAlmostEqual(float(call["benchmark"]) - float(put["benchmark"]), expected, places=10)

    def test_nondividend_american_call_equals_european(self):
        for rate in [0, 0.05]:
            european = self.price("binomial_european", type="call", steps=400, rate=rate)[0]
            american = self.price("binomial_american", type="call", steps=400, rate=rate)[0]
            self.assertAlmostEqual(float(european["price"]), float(american["price"]), places=10)

    def test_american_put_bounds(self):
        for spot in [70, 100, 130]:
            european = float(self.price("binomial_european", spot=spot, steps=400)[0]["price"])
            american = float(self.price("binomial_american", spot=spot, steps=400)[0]["price"])
            self.assertGreaterEqual(american + 1e-10, max(100 - spot, european))
            self.assertLessEqual(american, 100)

    def test_scale_homogeneity(self):
        for method in ["binomial_european", "binomial_american", "lsm_american"]:
            low = self.price(method, spot=100, strike=100, paths=2000, train_paths=2000, steps=20)[0]
            high = self.price(method, spot=200, strike=200, paths=2000, train_paths=2000, steps=20)[0]
            self.assertAlmostEqual(2 * float(low["price"]), float(high["price"]), places=9)

    def test_monte_carlo_accuracy(self):
        for kind in ["call", "put"]:
            row = self.price("monte_carlo_european", type=kind, paths=100000)[0]
            self.assertLess(abs(float(row["price"]) - float(row["benchmark"])),
                            6 * float(row["standard_error"]))

    def test_variance_estimators_and_work_accounting(self):
        rows = {row["method"]: row for row in self.price(
            "variance_reduction", type="call", paths=100000, train_paths=2000)}
        self.assertEqual(int(rows["antithetic"]["independent_samples"]), 50000)
        self.assertEqual(int(rows["plain_mc"]["independent_samples"]), 100000)
        self.assertEqual(int(rows["antithetic"]["payoff_evaluations"]), 100000)
        self.assertEqual(int(rows["control_variate"]["payoff_evaluations"]), 102000)
        self.assertEqual(int(rows["importance_sampling"]["payoff_evaluations"]), 102000)
        self.assertEqual(int(rows["importance_sampling"]["training_paths"]), 2000)
        for row in rows.values():
            self.assertLess(abs(float(row["price"]) - float(row["benchmark"])),
                            6 * float(row["standard_error"]))
        for method in ["antithetic", "control_variate", "importance_sampling"]:
            self.assertLess(float(rows[method]["standard_error"]), float(rows["plain_mc"]["standard_error"]))

    def test_newton_convergence_trace(self):
        for kind in ["call", "put"]:
            with TemporaryDirectory(prefix="newton-test-") as directory:
                trace_path = Path(directory) / "trace.csv"
                rows = self.price("variance_reduction", type=kind, paths=2000,
                                  train_paths=5000, newton_trace=trace_path)
                result = next(row for row in rows if row["method"] == "importance_sampling")
                with trace_path.open(newline="", encoding="utf-8") as source:
                    trace = list(csv.DictReader(source))
                self.assertIn(result["newton_status"], ["converged_gradient", "converged_step"])
                self.assertEqual(len(trace), int(result["newton_iterations"]) + 1)
                self.assertEqual(trace[-1]["status"], result["newton_status"])
                stopping_field = "gradient" if result["newton_status"] == "converged_gradient" else "step"
                self.assertLess(abs(float(trace[-1][stopping_field])), 1e-8)
                self.assertEqual(float(trace[-1]["shift"]), float(result["importance_shift"]))
                self.assertTrue(all(float(step["hessian"]) > 0.0 for step in trace))
                self.assertTrue(all(abs(float(step["step"])) <= 1.0 for step in trace))
                self.assertAlmostEqual(float(trace[0]["shift"]), 0.0 if kind == "call" else -0.15)
                # The first correction must be Newton on the raw second moment.
                initial = trace[0]
                correction = max(-1.0, min(1.0, float(initial["gradient"]) / float(initial["hessian"])))
                self.assertAlmostEqual(float(trace[1]["shift"]), float(initial["shift"]) - correction)
                objectives = [float(step["second_moment"]) for step in trace]
                self.assertLess(objectives[-1], objectives[0])
                self.assertTrue(all(b <= a + 1e-12 for a, b in zip(objectives, objectives[1:])))
                shift_sign = 1 if kind == "call" else -1
                self.assertGreater(shift_sign * float(result["importance_shift"]), 0)

    def test_shift_fitting_does_not_depend_on_evaluation_budget(self):
        results = []
        for paths in [2000, 10000]:
            rows = self.price("variance_reduction", paths=paths, train_paths=2000)
            results.append(next(row for row in rows if row["method"] == "importance_sampling"))
        for field in ["importance_shift", "newton_iterations", "newton_gradient",
                      "newton_second_moment_initial", "newton_second_moment_final"]:
            self.assertEqual(results[0][field], results[1][field])

    def test_newton_shift_against_independent_quadrature(self):
        # Validate the sampled optimizer against the population objective by quadrature,
        # without using its derivatives, stored pilot draws, or Newton iteration.
        for kind in ["call", "put"]:
            rows = self.price("variance_reduction", type=kind, paths=10000, train_paths=100000)
            row = next(result for result in rows if result["method"] == "importance_sampling")
            direction = 1 if kind == "call" else -1
            dz = 20 / 10000
            samples = []
            for i in range(10001):
                z = -10 + i * dz
                stock = 100 * math.exp(0.03 + 0.2 * z)
                payoff = math.exp(-0.05) * max(direction * (stock - 100), 0)
                trapezoid = 0.5 if i in [0, 10000] else 1
                samples.append((z, trapezoid * payoff ** 2 * math.exp(-0.5 * z * z) * dz /
                                math.sqrt(2 * math.pi)))

            def second_moment(shift):
                return sum(weight * math.exp(0.5 * shift * shift - shift * z)
                           for z, weight in samples)

            # Golden-section search is independent of the C++ Newton solver.
            left, right = (-4.0, 0.0) if kind == "put" else (0.0, 4.0)
            ratio = (math.sqrt(5) - 1) / 2
            for _ in range(50):
                lower = right - ratio * (right - left)
                upper = left + ratio * (right - left)
                if second_moment(lower) < second_moment(upper):
                    right = upper
                else:
                    left = lower
            population_optimum = 0.5 * (left + right)
            self.assertLess(abs(float(row["importance_shift"]) - population_optimum), 0.04)
            previous_shift = 0.35 if kind == "call" else -0.65
            self.assertLess(second_moment(float(row["importance_shift"])), second_moment(previous_shift))

    def test_no_positive_importance_pilot_fails_explicitly(self):
        result = subprocess.run([str(executable("variance_reduction", BUILD_DIR)),
                                "--type", "call", "--spot", "1", "--volatility", "0.01",
                                "--paths", "100", "--train-paths", "100"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pilot contains no positive payoffs", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_deterministic_importance_skips_pilot_and_optimizer(self):
        rows = self.price("variance_reduction", volatility=0, spot=80, paths=100, train_paths=100)
        row = next(result for result in rows if result["method"] == "importance_sampling")
        self.assertEqual(row["newton_status"], "deterministic")
        self.assertEqual(int(row["newton_iterations"]), 0)
        self.assertEqual(int(row["training_paths"]), 0)
        self.assertEqual(int(row["payoff_evaluations"]), 100)

    def test_newton_trace_cannot_overwrite_pricing_output(self):
        with TemporaryDirectory(prefix="newton-output-test-") as directory:
            path = str(Path(directory) / "same.csv")
            result = subprocess.run([str(executable("variance_reduction", BUILD_DIR)),
                                    "--paths", "100", "--train-paths", "100",
                                    "--output", path, "--newton-trace", path],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be different files", result.stderr)
            self.assertFalse(Path(path).exists())
    def test_importance_sampling_otm_call_and_put(self):
        for kind, spot in [("call", 80), ("put", 120)]:
            rows = self.price("variance_reduction", type=kind, spot=spot,
                              paths=100000, train_paths=2000)
            row = next(r for r in rows if r["method"] == "importance_sampling")
            self.assertLess(abs(float(row["price"]) - float(row["benchmark"])),
                            6 * float(row["standard_error"]))

    def test_fixed_policy_bias_does_not_vanish_with_paths(self):
        row = self.price("monte_carlo_american", boundary_ratio=0.01, paths=100000, steps=20)[0]
        # A near-zero put boundary almost never triggers: value is essentially European.
        self.assertLess(abs(float(row["price"]) - 5.573526022256971),
                        6 * float(row["standard_error"]))
        self.assertLess(float(row["price"]) + 3 * float(row["standard_error"]), float(row["benchmark"]))

    def test_lsm_close_to_american_tree_reference(self):
        row = self.price("lsm_american", paths=100000, train_paths=50000, steps=50)[0]
        tolerance = 6 * float(row["standard_error"]) + 0.12
        self.assertLess(abs(float(row["price"]) - float(row["benchmark"])), tolerance)
        self.assertGreater(float(row["price"]), 5.573526022256971)

    def test_paper_lsm_uses_antithetic_laguerre_in_sample_mode(self):
        row = self.price("lsm_american", lsm_mode="paper", type="put", spot=40, strike=40,
                         rate=0.06, volatility=0.2, maturity=1, steps=50,
                         paths=10000, train_paths=5000)[0]
        self.assertEqual(int(row["paths"]), 10000)
        self.assertEqual(int(row["training_paths"]), 10000)
        self.assertEqual(int(row["independent_samples"]), 5000)
        self.assertEqual(int(row["payoff_evaluations"]), 10000)
        self.assertLess(abs(float(row["price"]) - 2.313), 6 * float(row["standard_error"]) + 0.06)
        continuous = self.price("binomial_american", spot=40, strike=40, rate=0.06,
                                volatility=0.2, maturity=1, steps=1000)[0]
        self.assertLess(float(row["benchmark"]), float(continuous["price"]))

    def test_paper_lsm_requires_even_antithetic_path_count(self):
        result = subprocess.run([str(executable("lsm_american", BUILD_DIR)),
                                 "--lsm-mode", "paper", "--paths", "101"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("even --paths", result.stderr)

    def test_lsm_deep_itm_exercises_at_zero(self):
        row = self.price("lsm_american", spot=40, paths=1000, train_paths=1000, steps=20)[0]
        self.assertAlmostEqual(float(row["price"]), 60.0, places=10)
        self.assertEqual(float(row["standard_error"]), 0.0)
        self.assertEqual(float(row["exercise_fraction"]), 1.0)

    def test_lsm_no_itm_regression_is_finite(self):
        row = self.price("lsm_american", spot=1000, paths=1000, train_paths=1000,
                         volatility=0.01, steps=20)[0]
        self.assertEqual(float(row["price"]), 0.0)

    def test_lsm_one_exercise_interval(self):
        row = self.price("lsm_american", steps=1, paths=100000, train_paths=10000)[0]
        self.assertLess(abs(float(row["price"]) - 5.573526022256971),
                        6 * float(row["standard_error"]))

    def test_negative_rates(self):
        row = self.price("monte_carlo_european", rate=-0.02, paths=100000)[0]
        self.assertLess(abs(float(row["price"]) - float(row["benchmark"])),
                        6 * float(row["standard_error"]))
        # The no-early-exercise call shortcut must not be used for negative rates.
        row = self.price("lsm_american", type="call", rate=-0.05, spot=200,
                         volatility=0, paths=1000, train_paths=1000, steps=20)[0]
        self.assertAlmostEqual(float(row["price"]), 100, places=9)

    def test_zero_volatility(self):
        expected_european = 100 * math.exp(-0.05) - 80
        for method in METHODS:
            for row in self.price(method, volatility=0, spot=80, paths=1000, train_paths=1000, steps=20):
                expected = expected_european if method in ["binomial_european", "monte_carlo_european",
                                                          "variance_reduction"] else 20
                self.assertAlmostEqual(float(row["price"]), expected, places=9)
                self.assertLess(float(row["standard_error"]), 1e-10)

    def test_expired_contracts(self):
        for method in METHODS:
            for row in self.price(method, maturity=0, spot=90, paths=100, train_paths=100, steps=10):
                self.assertEqual(float(row["price"]), 10)
                self.assertEqual(float(row["standard_error"]), 0)

    def test_reproducibility(self):
        for method in ["monte_carlo_european", "monte_carlo_american", "variance_reduction", "lsm_american"]:
            first = self.price(method, paths=1000, train_paths=1000, steps=10)
            second = self.price(method, paths=1000, train_paths=1000, steps=10)
            for a, b in zip(first, second):
                self.assertEqual({k: v for k, v in a.items() if k != "runtime_ms"},
                                 {k: v for k, v in b.items() if k != "runtime_ms"})

    def test_invalid_inputs_fail_cleanly(self):
        for method in METHODS:
            for args in [["--spot", "-1"], ["--volatility", "nan"], ["--maturity", "-1"],
                         ["--type", "invalid"], ["--steps", "0"], ["--paths", "1"],
                         ["--seed", "-1"], ["--seed", "18446744073709551616"],
                         ["--unknown", "1"], ["--spot"]]:
                with self.subTest(method=method, args=args):
                    result = subprocess.run([str(executable(method, BUILD_DIR)), *args],
                                            capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Error:", result.stderr)
                    self.assertEqual(result.stdout, "")

    def test_invalid_probability_and_odd_antithetic_budget(self):
        for method, args in [("binomial_european", ["--rate", "1", "--steps", "1"]),
                             ("variance_reduction", ["--paths", "101"])]:
            result = subprocess.run([str(executable(method, BUILD_DIR)), *args],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--build-dir", type=Path, default=BUILD_DIR)
    options, remaining = parser.parse_known_args()
    BUILD_DIR = options.build_dir.resolve()
    unittest.main(argv=[sys.argv[0], *remaining])
