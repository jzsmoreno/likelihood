"""Test that the package passes pydocstyle with the numpy convention.

This test runs ``python -m pydocstyle likelihood --convention=numpy`` and
parses pydocstyle's output to identify undocumented classes and functions.

pydocstyle reports missing docstrings with the D1xx codes:

D100 - Missing docstring in public module
D101 - Missing docstring in public class
D102 - Missing docstring in public method
D103 - Missing docstring in public function

Run with::

    python -m pytest tests/test_linting_docstrings.py -v
"""

import ast
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_pydocstyle():
    """Run pydocstyle over the package and return the captured output."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pydocstyle",
            "likelihood",
            "--convention=numpy",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _get_enclosing_class(path, lineno):
    """Return the class containing a given line number, if any."""
    try:
        source = Path(path).read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None

    enclosing_classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            end_lineno = getattr(node, "end_lineno", node.lineno)

            if node.lineno <= lineno <= end_lineno:
                enclosing_classes.append(node)

    if not enclosing_classes:
        return None

    # Select the innermost enclosing class.
    return min(
        enclosing_classes,
        key=lambda node: getattr(node, "end_lineno", node.lineno) - node.lineno,
    )


def _extract_missing_docstrings(output):
    """Parse pydocstyle output and return the undocumented classes and functions.

    pydocstyle prints each violation as two consecutive lines::

        <file>:<line> [<in public class ``Name`` | at module level]:
                D###: <message>

    The header line (carrying the name in backticks) immediately precedes the
    indented ``D###`` message line. Each D1xx message line counts as one
    undocumented entity: D100 (module), D101 (class), D102 (method), D103
    (function). A D102 message is attached to the nearest preceding D101 class
    header in the same file.
    """
    lines = output.splitlines()

    undocumented = {
        "modules": [],
        "classes": [],
        "functions": [],
    }

    for i, line in enumerate(lines):
        code_match = re.search(r"\b(D100|D101|D102|D103):", line)

        if not code_match:
            continue

        code = code_match.group(1)

        if i == 0:
            continue

        header = lines[i - 1]

        # Extract file path and line number.
        location_match = re.match(r"(.+?):(\d+)", header)

        if not location_match:
            continue

        path = location_match.group(1)
        lineno = int(location_match.group(2))

        # Extract the entity name from the pydocstyle header.
        name_match = re.search(r"`([^`]*)`", header)
        name = name_match.group(1) if name_match else ""

        if code == "D100":
            undocumented["modules"].append(name)

        elif code == "D101":
            undocumented["classes"].append(
                {
                    "name": name,
                    "path": path,
                    "lineno": lineno,
                    "methods": [],
                }
            )

        elif code == "D102":
            enclosing_class = _get_enclosing_class(path, lineno)

            if enclosing_class is not None:
                class_entry = next(
                    (
                        entry
                        for entry in undocumented["classes"]
                        if entry["path"] == path and entry["name"] == enclosing_class.name
                    ),
                    None,
                )

                if class_entry is None:
                    class_entry = {
                        "name": enclosing_class.name,
                        "path": path,
                        "lineno": enclosing_class.lineno,
                        "methods": [],
                    }
                    undocumented["classes"].append(class_entry)

                class_entry["methods"].append(name)
            else:
                undocumented["functions"].append(name)

        elif code == "D103":
            undocumented["functions"].append(name)

    return undocumented


def _count_missing_docstrings(undocumented):
    """Return the total number of missing docstrings."""
    return (
        len(undocumented["modules"])
        + len(undocumented["classes"])
        + sum(len(klass["methods"]) for klass in undocumented["classes"])
        + len(undocumented["functions"])
    )


def _assert_pydocstyle_succeeded(result):
    """Ensure pydocstyle executed successfully."""
    assert result.returncode == 0, (
        "pydocstyle execution failed.\n" f"stdout:\n{result.stdout}\n" f"stderr:\n{result.stderr}"
    )


def test_no_missing_public_docstrings():
    """The package should have no public entities missing docstrings."""
    result = run_pydocstyle()
    _assert_pydocstyle_succeeded(result)

    undocumented = _extract_missing_docstrings(result.stdout)

    assert not undocumented["classes"], f"Undocumented classes: {undocumented['classes']}"

    assert not undocumented["functions"], f"Undocumented functions: {undocumented['functions']}"

    assert not undocumented["modules"], f"Undocumented modules: {undocumented['modules']}"

    for klass in undocumented["classes"]:
        assert not klass["methods"], (
            f"Undocumented methods in {klass['name']}: " f"{klass['methods']}"
        )


def test_parser_recovers_all_missing_docstrings():
    """The parser should recover all D100-D103 violations."""
    result = run_pydocstyle()
    _assert_pydocstyle_succeeded(result)

    pydocstyle_missing = [
        line for line in result.stdout.splitlines() if re.search(r"\b(D100|D101|D102|D103):", line)
    ]

    pydocstyle_count = len(pydocstyle_missing)

    undocumented = _extract_missing_docstrings(result.stdout)
    parser_count = _count_missing_docstrings(undocumented)

    assert parser_count == pydocstyle_count, (
        f"Parser found {parser_count} missing docstrings, " f"pydocstyle found {pydocstyle_count}"
    )


def test_parser_handles_empty_output():
    """The parser should return empty results for empty output."""
    result = _extract_missing_docstrings("")

    assert result == {
        "modules": [],
        "classes": [],
        "functions": [],
    }


def test_parser_extracts_missing_module_docstring():
    """The parser should identify undocumented public modules."""
    output = (
        "likelihood/example.py:1 at module level:\n"
        "        D100: Missing docstring in public module\n"
    )

    result = _extract_missing_docstrings(output)

    assert len(result["modules"]) == 1


def test_parser_extracts_missing_class_docstring():
    """The parser should identify undocumented public classes."""
    output = (
        "likelihood/example.py:10 in public class `Example`:\n"
        "        D101: Missing docstring in public class\n"
    )

    result = _extract_missing_docstrings(output)

    assert len(result["classes"]) == 1
    assert result["classes"][0]["name"] == "Example"


def test_parser_extracts_missing_function_docstring():
    """The parser should identify undocumented public functions."""
    output = (
        "likelihood/example.py:10 at module level:\n"
        "        D103: Missing docstring in public function\n"
    )

    result = _extract_missing_docstrings(output)

    assert len(result["functions"]) == 1
