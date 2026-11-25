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
    version = subprocess.run([str(compiler), "--version"], capture_output=True, text=True, check=True)
    sources = sorted((ROOT / "src").glob("*.cpp")) + sorted((ROOT / "include").glob("*.hpp"))
    metadata = {"compiler": version.stdout.splitlines()[0], "cpp_standard": "c++17",
                "optimization": "-O0 -g" if debug else "-O2",
                "source_sha256": {path.relative_to(ROOT).as_posix():
                                  hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}
    (build_dir / "build_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    try:
        build(args.compiler, args.build_dir, args.debug)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")
