"""Automated tests for Jupyter notebooks in the examples/ directory.

This module uses pytest and nbclient to discover all .ipynb files under the
examples/ directory, execute them programmatically into a temporary directory
created with tempfile.TemporaryDirectory(), and verify that they run without
exceptions.

Requirements:
    - pytest
    - jupyter (for nbconvert)
    - nbconvert
    - nbclient

Run with increased verbosity for easier debugging:

pytest -vv -s
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

print(f"repo_root = {repo_root}")
print(f"sys.path[0] = {sys.path[0]}")
print(f"Existe likelihood: {(repo_root / 'likelihood').exists()}")
print(f"Existe __init__.py: {(repo_root / 'likelihood' / '__init__.py').exists()}")

from pathlib import Path
from shutil import copy2, copytree
from tempfile import TemporaryDirectory

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

import likelihood

print("\n" + "=" * 80)
print("Repository package")
print("=" * 80)
print(f"Version : {likelihood.__version__}")
print(f"Location: {likelihood.__file__}")
print("=" * 80)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

SKIP_NOTEBOOKS = {
    "AutoClassifier with Optuna.ipynb",
}


def find_notebooks():
    notebooks = sorted(EXAMPLES_DIR.glob("*.ipynb"))
    params = []

    for notebook in notebooks:
        if notebook.name in SKIP_NOTEBOOKS:
            params.append(
                pytest.param(
                    notebook,
                    marks=pytest.mark.skip(reason="Notebook excluded from CI"),
                    id=notebook.stem,
                )
            )
        else:
            params.append(
                pytest.param(
                    notebook,
                    id=notebook.stem,
                )
            )

    return params


@pytest.mark.parametrize("notebook_path", find_notebooks())
def test_notebook_execution(notebook_path: Path):
    """Execute a notebook and verify that it runs without errors."""

    with notebook_path.open(encoding="utf-8") as fh:
        nb = nbformat.read(fh, as_version=4)

    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
    )

    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()

        for item in EXAMPLES_DIR.iterdir():
            if item.is_file() and item.suffix != ".ipynb":
                copy2(item, output_dir / item.name)

        copytree(
            repo_root / "likelihood",
            output_dir / "likelihood",
            dirs_exist_ok=True,
        )

        try:
            client.execute(cwd=output_dir)

        except CellExecutionError as exc:
            if "URLError" in str(exc) or "getaddrinfo failed" in str(exc):
                pytest.skip("The notebook requires Internet access")
            print("\n" + "=" * 80)
            print(f"Notebook failed: {notebook_path.name}")
            print("=" * 80)

            print("\nError details:")
            print(exc)

            if hasattr(exc, "cell"):
                cell = exc.cell
                print("\nSource code of the failing cell:")
                print("-" * 80)
                print(cell.get("source", ""))
                print("-" * 80)

                print("\nOutputs:")
                for output in cell.get("outputs", []):
                    print(output)

            raise
