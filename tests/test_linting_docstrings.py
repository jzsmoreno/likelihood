"""Test that the package passes pydocstyle with the numpy convention.

This test runs ``python -m pydocstyle likelihood --convention=numpy`` and
parses pydocstyle's output to surface undocumented classes and functions.

pydocstyle reports missing docstrings with the D1xx codes:

D100 - Missing docstring in public module
D101 - Missing docstring in public class
D102 - Missing docstring in public method
D103 - Missing docstring in public function

Run with::

    python -m pytest tests/test_linting_docstrings.py -v
"""

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_pydocstyle():
    """Run pydocstyle over the package and return the captured output."""
    result = subprocess.run(
        [sys.executable, "-m", "pydocstyle", "likelihood", "--convention=numpy"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    return result


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
    # classes_by_line maps the index of each D101 header line to its class entry,
    # so a D102 method can be attached to the class whose header precedes it.
    classes_by_line = {}
    # Walk the lines once. For every D1xx message line, the header that immediately
    # precedes it (one line up) carries the entity name. We bucket by code and, for
    # D102, attach the method to the most recent D101 header seen in the same file.
    last_class = None
    last_class_name = None
    for i, line in enumerate(lines):
        code_match = re.search(r"\s(D100|D101|D102|D103):", line)
        if not code_match:
            continue
        code = code_match.group(1)
        # Header line immediately precedes the message line.
        header = lines[i - 1]
        name_match = re.search(r"`([^`]*)`", header)
        name = name_match.group(1) if name_match else ""
        if code == "D101":
            entry = {"name": name, "path": "", "lineno": 0, "methods": []}
            classes_by_line[i - 1] = entry
            last_class = entry
            last_class_name = name
            undocumented["classes"].append(entry)
        elif code == "D103":
            undocumented["functions"].append(name)
        elif code == "D100":
            undocumented["modules"].append(name)
        elif code == "D102":
            if last_class is not None:
                last_class["methods"].append(name)
    return undocumented


def test_no_undocumented_classes_or_functions():
    """The package should have no classes or functions missing a docstring."""
    result = run_pydocstyle()
    undocumented = _extract_missing_docstrings(result.stdout)

    assert not undocumented["classes"], f"Undocumented classes: {undocumented['classes']}"
    assert not undocumented["functions"], f"Undocumented functions: {undocumented['functions']}"
    assert not undocumented["modules"], f"Undocumented modules: {undocumented['modules']}"
    for klass in undocumented["classes"]:
        assert not klass["methods"], f"Undocumented methods in {klass['name']}: {klass['methods']}"


def test_parser_recovers_all_missing_docstrings():
    """The parser should recover the exact number of undocumented entities."""
    result = run_pydocstyle()

    pydocstyle_missing = [
        line for line in result.stdout.splitlines() if re.search(r"(D100|D101|D102|D103):", line)
    ]
    pydocstyle_count = len(pydocstyle_missing)

    undocumented = _extract_missing_docstrings(result.stdout)
    parser_count = (
        len(undocumented["classes"])
        + sum(len(klass["methods"]) for klass in undocumented["classes"])
        + len(undocumented["functions"])
        + len(undocumented["modules"])
    )

    assert (
        parser_count == pydocstyle_count
    ), f"parser found {parser_count} missing docstrings, pydocstyle found {pydocstyle_count}"
    assert parser_count > 0, "Expected the parser to recover undocumented entities"
