"""Research-file protection and source-aware CRR cache tests (standard library)."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_files import prepare_results, plot_locations, results_dir, REFERENCE_DIR
from crr_cache import cache_key, request_for, get_reference, source_hashes

BUILD_DIR = ROOT / "build"
CONTRACT = dict(spot=40, strike=40, rate=.06, volatility=.2, maturity=1, type="put")


class ResearchFilesTests(unittest.TestCase):
    def test_study_and_smoke_defaults_are_separate(self):
        self.assertNotEqual(results_dir("paper_replication"), results_dir("lsm_comparison"))
        self.assertTrue(results_dir("paper_replication", True).is_relative_to(ROOT / "tmp"))

    def test_smoke_cannot_target_research_data_or_figures(self):
        for path in (ROOT / "data/experiments/paper_replication/full", ROOT / "figures/paper_replication/full",
                     REFERENCE_DIR, ROOT / "archive/new_smoke"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                prepare_results("paper_replication", path, True)

    def test_existing_research_and_partial_results_are_not_overwritten(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "manifest.json").write_text(json.dumps({"profile": "full"}))
            with self.assertRaises(ValueError): prepare_results("lsm_comparison", directory, True)
            with self.assertRaises(ValueError): prepare_results("lsm_comparison", directory, False)
            (directory / "manifest.json").unlink()
            (directory / "partial.csv").write_text("preserve me")
            with self.assertRaises(ValueError): prepare_results("lsm_comparison", directory, False)
            self.assertEqual((directory / "partial.csv").read_text(), "preserve me")

    def test_smoke_figures_cannot_overwrite_full_figures(self):
        with TemporaryDirectory() as temp:
            data = Path(temp)
            (data / "manifest.json").write_text(json.dumps({"profile": "smoke_only"}))
            with self.assertRaises(ValueError):
                plot_locations("lsm_comparison", data, ROOT / "figures/lsm_comparison/full")
            _, figures = plot_locations("lsm_comparison", data)
            self.assertEqual(figures, data / "figures")

    def test_reference_bundle_matches_its_provenance(self):
        provenance = json.loads((REFERENCE_DIR / "provenance.json").read_text())
        for table in provenance["tables"].values():
            self.assertEqual(hashlib.sha256((REFERENCE_DIR / table["file"]).read_bytes()).hexdigest(), table["sha256"])

    def test_cache_changes_with_every_contract_parameter_and_grid(self):
        original = cache_key(request_for(CONTRACT, 200, 50))
        for name, value in dict(spot=42, strike=41, rate=.07, volatility=.3, maturity=2, type="call").items():
            changed = dict(CONTRACT); changed[name] = value
            self.assertNotEqual(original, cache_key(request_for(changed, 200, 50)))
        self.assertNotEqual(original, cache_key(request_for(CONTRACT, 400, 50)))
        self.assertNotEqual(original, cache_key(request_for(CONTRACT, 200, 100)))

    def test_cache_changes_with_relevant_source_code(self):
        original = cache_key(request_for(CONTRACT, 200, 50))
        hashes = source_hashes(); hashes["include/binomial.hpp"] = "changed"
        with patch("crr_cache.source_hashes", return_value=hashes):
            self.assertNotEqual(original, cache_key(request_for(CONTRACT, 200, 50)))

    def test_cpp_cache_miss_hit_and_corruption_recovery(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            first = get_reference(BUILD_DIR, CONTRACT, 200, 50, directory)
            self.assertFalse(first["cache_hit"])
            second = get_reference(BUILD_DIR, CONTRACT, 200, 50, directory)
            self.assertTrue(second["cache_hit"])
            self.assertEqual(first["american_grid_reference_crr"], second["american_grid_reference_crr"])
            self.assertGreaterEqual(second["american_reference_crr"], second["american_grid_reference_crr"])
            (directory / (first["cache_key"] + ".json")).write_text("broken")
            self.assertFalse(get_reference(BUILD_DIR, CONTRACT, 200, 50, directory)["cache_hit"])

    def test_source_change_rejects_stale_executable_instead_of_using_cached_price(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            get_reference(BUILD_DIR, CONTRACT, 200, 50, directory)
            hashes = source_hashes(); hashes["include/model.hpp"] = "changed"
            with patch("crr_cache.source_hashes", return_value=hashes), self.assertRaisesRegex(RuntimeError, "stale"):
                get_reference(BUILD_DIR, CONTRACT, 200, 50, directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--build-dir", type=Path, default=BUILD_DIR)
    args, remaining = parser.parse_known_args()
    BUILD_DIR = args.build_dir.resolve()
    unittest.main(argv=[__file__, *remaining])
