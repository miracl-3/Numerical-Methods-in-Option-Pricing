"""Compile and run deterministic checks against the production LSM functions."""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = None


class LsmProcedureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            raise unittest.SkipTest("Procedure fixtures require g++ or clang++.")
        cls.directory = tempfile.TemporaryDirectory(prefix="lsm-procedure-")
        cls.binary = Path(cls.directory.name) / "lsm_procedure.exe"
        result = subprocess.run([
            compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic",
            "-I", str(ROOT / "src"), "-I", str(ROOT / "include"),
            str(ROOT / "tests" / "lsm_procedure.cpp"), "-o", str(cls.binary),
        ], capture_output=True, text=True)
        if result.returncode:
            cls.directory.cleanup()
            raise RuntimeError(result.stderr)
        if EXAMPLE_DIR is not None:
            result = subprocess.run([str(cls.binary), "example", str(EXAMPLE_DIR)],
                                    capture_output=True, text=True, check=True)
            print(result.stdout.strip())

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def fixture(self, name):
        result = subprocess.run([str(self.binary), name], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_original_eight_path_stopping_rule(self): self.fixture("published")
    def test_pair_and_naive_standard_error_accounting(self): self.fixture("standard_errors")
    def test_cashflow_normalization_and_exercise_at_zero(self): self.fixture("time_zero")
    def test_antithetic_paths_and_exact_gbm(self): self.fixture("paths")
    def test_normalization_and_scale_homogeneity(self): self.fixture("scale")
    def test_reuse_of_normalized_coefficients(self): self.fixture("reuse")
    def test_backward_realized_cashflow_updating(self): self.fixture("backward")
    def test_no_itm_insufficient_itm_and_rank_deficiency(self): self.fixture("degeneracy")
    def test_qr_fit_and_conditioning_diagnostic(self): self.fixture("qr")
    def test_frozen_policy_matches_training_decisions(self): self.fixture("frozen")
    def test_sampling_and_valuation_metadata(self): self.fixture("metadata")
    def test_exercise_grid_matches_crr(self): self.fixture("grid")
    def test_independent_evaluation_does_not_rechoose_time_zero(self): self.fixture("no_test_max")
    def test_reproducible_streams_and_plain_odd_paths(self): self.fixture("reproducible")
    def test_deterministic_expiry_and_nondividend_call(self): self.fixture("deterministic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--example-dir", type=Path)
    args, remaining = parser.parse_known_args()
    EXAMPLE_DIR = args.example_dir.resolve() if args.example_dir else None
    unittest.main(argv=[__file__, *remaining])
