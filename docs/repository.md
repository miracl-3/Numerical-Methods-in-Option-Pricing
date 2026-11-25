# Repository organization and file management

`M` denotes maintained source/text, `G` generated research evidence, `F` frozen
historical material, and `L` ignored local output.

```text
README.md, requirements*.txt, CMakeLists.txt       M  setup and build configuration
.github/, .vscode/, .clang-format                 M  automation and editor/style configuration
src/                                             M  six C++ pricing entry points
include/                                         M  shared model, CRR and CSV interface
scripts/
  build.py                                       M  primary compilation and build hashes
  experiment_files.py                            M  study paths and overwrite rules
  crr_cache.py                                   M  cache of C++ CRR output
  run_*.py, plot_*.py                             M  orchestration, statistics and plots
  verify_notebook.py                             M  optional execution to a temporary copy
tests/                                           M  numerical, procedure and file/cache validation
data/
  reference/longstaff_schwartz_2001/
    table1.csv, table2.csv                        M  manually verified published tables
    provenance.json                              M  canonical source record and table hashes
  experiments/
    paper_replication/full/                      G  original-paper runs, SE summaries and manifest
    lsm_comparison/full/                         G  plain/AV fits, Table 2, QR diagnostics and manifest
      example/                                   G  deterministic coefficients and decisions
    pricing_methods/historical_quick/             G  preserved earlier broader study
    STUDY/NEW_RUN/                                G  later explicitly named research runs
figures/STUDY/RUN/                                G  PNG/SVG figures corresponding to saved data
docs/
  methodology.md                                 M  equations, function map, complexity and example
  repository.md                                  M  file ownership and lifecycle
  results/                                       M  numerical findings and limitations
notebooks/option_pricing.ipynb                    M/G optional source cells and saved outputs
extensions/importance_sampling_2009/              F  distinct historical method and results
archive/
  organization_2026-11-26/                        F  relocation hashes, old guides and supporting files
  early_implementations/                          F  source/result snapshots from local recovery ZIPs
  legacy_monte_carlo.zip                          F  six preserved earlier simulation CSVs
.venv/                                           L  one active Python environment
build/                                           L  executables, optional CMake build and build manifest
.cache/crr/                                      L  validated, rebuildable CRR cache
.cache/                                          L  package/interpreter/notebook caches
tmp/smoke/STUDY/, tmp/notebooks/                   L  smoke runs and executed notebook copies
data/custom/                                     L  explicitly requested ad-hoc output
.local-archive/                                  L  retained original machine-local recovery ZIPs
```

## Maintained versus generated

Maintain algorithms, experiment scripts, tests, configuration, published-table
transcriptions and explanatory documents. Generate experimental CSVs, statistical
summaries, manifests and figures from C++. Saved research outputs are intentionally
versioned; do not correct their prices by hand. Fix and test the source, then use
a fresh run directory. Reports interpret the cited saved outputs.

There is one canonical published-data bundle. Tables 1 and 2 have different
schemas and remain separate CSVs with one provenance record. Published values
copied into result CSVs are derived comparison columns, not another authority.

Paper replication and the plain/AV experiment remain separate. The broader
earlier study is honestly labeled `historical_quick`, not a full-budget run.
Individual estimates, summaries and original manifests are retained.

## Protection and provenance

Full runners refuse nonempty output directories, including partial failed runs.
Choose a fresh `--output-dir`. There is no automatic deletion or overwrite flag.
`--quick` goes to `tmp/smoke/STUDY/` and rejects destinations under research data,
figures, references and archives. Plots inspect the input manifest and keep smoke
figures temporary. Full plots map `data/experiments/STUDY/RUN/` to
`figures/STUDY/RUN/`.

Historical manifests were preserved byte-for-byte, including their original
source hashes and paths. [The relocation inventory](../archive/organization_2026-11-26/inventory.json)
resolves their old paths. Relocations and ZIP members were checked by SHA256;
their numerical contents were not rewritten.

## CRR cache

The JSON key includes put/call, all contract parameters, tree steps, matched
exercise-date count, time-zero convention and SHA256 of the CRR entry point and
shared pricing/interface headers. A changed request is a miss. Shared-header
fingerprinting is conservative; unrelated LSM-source and plot edits do not
invalidate it, but an interface-header edit can.

A miss checks source and executable build hashes before calling C++. The CRR
entry point exports both exercise schedules through the same existing tree
algorithm. Python performs no backward induction. Entries retain values,
command, runtime, source/build details and executable hash; writes are atomic.
Corrupt entries are misses. Experiment manifests preserve the keys and values
used, so clearing the local cache does not erase research evidence.

Historical tables are not silently imported as current cache entries when
fingerprints differ. Statistical analysis may reuse independently audited saved
reference values with their original provenance. A cold current-source full run
may need to compute missing references once; later compatible runs reuse them.

## Redundancies reviewed

| Material and purpose | Decision |
|---|---|
| Three `include/*.exe`, 201 MiB | Remove verified GCC precompiled-header artifacts; retain headers and artifact hashes |
| Quickstart and separate LSM implementation guide | Archive originals; merge useful content into README and methodology |
| Earlier README/methodology | Preserve frozen copies for historical context |
| CV/application LaTeX text | Archive as supporting material, outside the methodology |
| Six loose Monte Carlo CSVs | Archive with hashes; earlier simulation evidence with incomplete producer provenance |
| Historical importance-sampling extension/private headers | Retain a frozen implementation and its results, outside the main build |
| Local recovery ZIPs | Preserve portable nonbinary source/result snapshots; retain original local ZIPs |
| Optional notebook | Retain; update paths and keep numerical calculations in C++ |
| Broken `.venv` and duplicate working environment | Verify one rebuilt `.venv` before removing the duplicate generated environment; retain inventories |
| Executables, bytecode and smoke/tool caches | Ignore and regenerate when needed |

Archive code is historical, not part of current validation. Preserving it does
not certify its numerical correctness. Current tests target `src/` and `include/`.

## Optional tools

Install `requirements-notebooks.txt` only when needed. `verify_notebook.py` writes
an executed copy to `tmp/notebooks/`, leaving the saved walkthrough intact.
VS Code uses the selected `.venv` interpreter. Core requirements remain NumPy,
pandas and Matplotlib; compilation and automated tests use the standard library.

This workspace uses a cached CPython base recorded in `.venv/pyvenv.cfg`.
Retain that base interpreter while using this environment; clearing the CRR
cache is different from removing the interpreter itself.

Optional CMake workflow:

```text
cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake --config Release
ctest --test-dir build/cmake -C Release --output-on-failure
```

CMake calls the same metadata writer after building. For experiment scripts,
pass `--build-dir build/cmake`, or `build/cmake/Release` with Visual Studio.
The primary workflow remains `scripts/build.py`.
