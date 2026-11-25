"""Financial identities, numerical accuracy, statistical accounting, and CLI validation."""

import argparse
import math
from pathlib import Path
import subprocess
import sys
import unittest

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
        for row in rows.values():
            self.assertLess(abs(float(row["price"]) - float(row["benchmark"])),
                            6 * float(row["standard_error"]))
        for method in ["antithetic", "control_variate"]:
            self.assertLess(float(rows[method]["standard_error"]), float(rows["plain_mc"]["standard_error"]))

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
