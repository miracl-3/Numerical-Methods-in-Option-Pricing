"""Build the six independent C++17 executables with GCC or Clang; no Python packages needed."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "binomial_european", "binomial_american", "monte_carlo_european",
    "monte_carlo_american", "variance_reduction", "lsm_american",
)


def executable(name, build_dir=ROOT / "build"):
    return Path(build_dir) / (name + (".exe" if sys.platform == "win32" else ""))


def record_metadata(compiler, build_dir, optimization):
    version = subprocess.run([str(compiler), "--version"], capture_output=True, text=True)
    description = (version.stdout or version.stderr).splitlines()
    sources = sorted((ROOT / "src").glob("*.cpp")) + sorted((ROOT / "include").glob("*.hpp"))
    metadata = {"compiler": description[0] if description else str(compiler),
                "cpp_standard": "c++17", "optimization": optimization,
                "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sources},
                "binary_sha256": {name: hashlib.sha256(executable(name, build_dir).read_bytes()).hexdigest()
                                  for name in METHODS}}
    (Path(build_dir) / "build_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def build(compiler=None, build_dir=ROOT / "build", debug=False):
    compiler = compiler or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        raise RuntimeError("Install g++ or clang++, or pass --compiler PATH. MSVC users: use CMake.")
    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    for name in METHODS:
        command = [str(compiler), "-std=c++17", "-Wall", "-Wextra", "-Wpedantic",
                   "-O0" if debug else "-O2", "-I", str(ROOT / "include"),
                   str(ROOT / "src" / f"{name}.cpp"), "-o", str(executable(name, build_dir))]
        if debug:
            command.insert(2, "-g")
        subprocess.run(command, cwd=ROOT, check=True)
        print(f"Built {name}", flush=True)
    record_metadata(compiler, build_dir, "-O0 -g" if debug else "-O2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--record-only", action="store_true", help="CMake post-build metadata; does not compile.")
    parser.add_argument("--optimization", default="CMake-selected")
    args = parser.parse_args()
    try:
        if args.record_only:
            record_metadata(args.compiler, args.build_dir, args.optimization)
        else:
            build(args.compiler, args.build_dir, args.debug)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")
