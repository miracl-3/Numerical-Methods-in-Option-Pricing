"""Execute the research notebook; write portable executed output back into the notebook."""

import argparse
from pathlib import Path
import os
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
from tempfile import TemporaryDirectory
import json

ROOT = Path(__file__).resolve().parents[1]


def execute(notebook):
    # Use this exact interpreter without registering a kernel in the user's profile.
    with TemporaryDirectory(prefix="pricing-kernel-") as directory:
        kernel = Path(directory) / "pricing-python"
        kernel.mkdir()
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Option pricing Python", "language": "python",
        }), encoding="utf-8")
        environment = os.environ.copy()
        environment["JUPYTER_RUNTIME_DIR"] = str(ROOT / ".cache" / "jupyter-runtime")
        environment["IPYTHONDIR"] = str(ROOT / ".cache" / "ipython")
        Path(environment["JUPYTER_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)
        Path(environment["IPYTHONDIR"]).mkdir(parents=True, exist_ok=True)
        document = nbformat.read(notebook, as_version=4)
        nbformat.validate(document)
        manager = KernelSpecManager(kernel_dirs=[directory])
        from jupyter_client import KernelManager
        kernel_manager = KernelManager(kernel_name="pricing-python", kernel_spec_manager=manager)
        client = NotebookClient(document, timeout=120, kernel_name="pricing-python", km=kernel_manager,
                                resources={"metadata": {"path": str(ROOT / "notebooks")}})
        client.execute(env=environment)
        nbformat.write(document, notebook)
    print(f"Executed and validated {notebook.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", type=Path,
                        default=ROOT / "notebooks" / "option_pricing.ipynb")
    options = parser.parse_args()
    execute(options.notebook)
