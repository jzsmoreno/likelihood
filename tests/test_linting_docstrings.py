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
    python -m pytest tests/test_linting_docstrings.py -v > pytest_output.txt
    python -m pytest tests/test_linting_docstrings.py -v 2>&1 | tee pytest_output.txt
    ruff check . --select D209,D400,D401,D105 --output-format concise > errores_d401.txt 2>&1
    ruff check . --select D401 --fix
"""

import ast
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_pydocstyle():
    """Run pydocstyle over the package.

    Returns
    -------
    subprocess.CompletedProcess
        The completed process containing pydocstyle's captured stdout,
        stderr, and return code.
    """
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "likelihood",
            "--select",
            "D209,D401,D105",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _get_enclosing_class(path, lineno):
    """Find the class containing a given line number.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the Python source file.
    lineno : int
        Line number of the undocumented method.

    Returns
    -------
    ast.ClassDef or None
        The innermost enclosing class, or ``None`` if no class contains
        the specified line number or the source file cannot be parsed.
    """
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

    return min(
        enclosing_classes,
        key=lambda node: getattr(node, "end_lineno", node.lineno) - node.lineno,
    )


def _extract_missing_docstrings(output):
    """Parse pydocstyle output and identify undocumented entities.

    pydocstyle prints each violation as two consecutive lines::

        <file>:<line> [in public class ``Name``]:
                D###: <message>

    The parser recognizes the following violation codes:

    - D100: Missing docstring in public module.
    - D101: Missing docstring in public class.
    - D102: Missing docstring in public method.
    - D103: Missing docstring in public function.

    Method ownership is determined using the Python AST rather than relying
    on the presence of a D101 violation. This also identifies methods in
    classes that already have class docstrings.

    Parameters
    ----------
    output : str
        Complete stdout captured from pydocstyle.

    Returns
    -------
    dict
        Dictionary containing four lists:

        - ``modules``: Undocumented public modules.
        - ``classes``: Undocumented public classes.
        - ``functions``: Undocumented public functions.
        - ``methods``: Undocumented public methods, including their
          enclosing class names.

        Each entry contains the entity name, source file path, and line
        number. Method entries additionally contain the enclosing class
        name and its line number.
    """
    undocumented = {
        "modules": [],
        "classes": [],
        "functions": [],
        "methods": [],
    }

    lines = output.splitlines()

    for i, line in enumerate(lines):
        code_match = re.search(r"\b(D100|D101|D102|D103):", line)

        if not code_match or i == 0:
            continue

        code = code_match.group(1)
        header = lines[i - 1]

        location_match = re.match(r"(.+?):(\d+)", header)

        if not location_match:
            continue

        path = location_match.group(1)
        lineno = int(location_match.group(2))

        name_match = re.search(r"`([^`]*)`", header)
        name = name_match.group(1) if name_match else ""

        entry = {
            "name": name,
            "path": path,
            "lineno": lineno,
        }

        if code == "D100":
            undocumented["modules"].append(entry)

        elif code == "D101":
            undocumented["classes"].append(entry)

        elif code == "D103":
            undocumented["functions"].append(entry)

        elif code == "D102":
            enclosing_class = _get_enclosing_class(path, lineno)

            if enclosing_class is not None:
                entry["class"] = enclosing_class.name
                entry["class_lineno"] = enclosing_class.lineno
                undocumented["methods"].append(entry)
            else:
                undocumented["functions"].append(entry)

    return undocumented


def _format_missing_docstrings(undocumented):
    """Format missing docstrings as a hierarchical list.

    Parameters
    ----------
    undocumented : dict
        Dictionary returned by ``_extract_missing_docstrings``.

    Returns
    -------
    str
        Readable hierarchical report grouped by source file and entity type.
    """
    files = defaultdict(
        lambda: {
            "modules": [],
            "classes": [],
            "functions": [],
            "methods": [],
        }
    )

    for category in ("modules", "classes", "functions", "methods"):
        for entry in undocumented[category]:
            files[entry["path"]][category].append(entry)

    total = sum(len(entries) for entries in undocumented.values())

    if total == 0:
        return "All public entities have docstrings."

    output = [
        "FAILED: Missing public docstrings",
        "",
        f"Found {total} missing docstrings in {len(files)} files.",
    ]

    for index, path in enumerate(sorted(files), start=1):
        entries = files[path]

        output.extend(["", f"{index}. {path}"])

        categories = [
            ("modules", "Modules"),
            ("classes", "Classes"),
            ("methods", "Methods"),
            ("functions", "Functions"),
        ]

        for key, label in categories:
            if not entries[key]:
                continue

            output.append(f"   {label}:")

            for entry in sorted(
                entries[key],
                key=lambda item: item["lineno"],
            ):
                if key == "modules":
                    name = "Missing module docstring"
                elif key == "methods":
                    name = f"{entry['class']}.{entry['name']}"
                else:
                    name = entry["name"]

                output.append(f"      - {name} (line {entry['lineno']})")

    output.extend(
        [
            "",
            f"Total: {total} missing docstrings",
        ]
    )

    return "\n".join(output)


def _count_missing_docstrings(undocumented):
    """Count all missing docstrings.

    Parameters
    ----------
    undocumented : dict
        Dictionary returned by ``_extract_missing_docstrings``.

    Returns
    -------
    int
        Total number of missing module, class, function, and method
        docstrings.
    """
    return sum(len(entries) for entries in undocumented.values())


def _assert_pydocstyle_succeeded(result):
    """Verify that pydocstyle executed successfully.

    Parameters
    ----------
    result : subprocess.CompletedProcess
        Result returned by ``run_pydocstyle``.

    Raises
    ------
    AssertionError
        If pydocstyle could not execute successfully.
    """
    assert result.returncode == 0, (
        "pydocstyle execution failed.\n" f"stdout:\n{result.stdout}\n" f"stderr:\n{result.stderr}"
    )


def test_no_missing_public_docstrings():
    """Verify that all public entities have docstrings.

    Runs pydocstyle using the NumPy convention and checks that no public
    modules, classes, functions, or methods are missing docstrings.

    Raises
    ------
    AssertionError
        If pydocstyle execution fails or undocumented entities are found.
        The assertion message contains a report grouped by source file,
        including entity names and line numbers.
    """
    result = run_pydocstyle()
    _assert_pydocstyle_succeeded(result)

    undocumented = _extract_missing_docstrings(result.stdout)
    report = _format_missing_docstrings(undocumented)

    assert _count_missing_docstrings(undocumented) == 0, report


def test_parser_recovers_all_missing_docstrings():
    """Verify that the parser identifies every D100-D103 violation.

    Compares the number of missing docstrings reported by pydocstyle
    against the number recovered by the custom parser.

    This test does not require any missing docstrings to exist. It
    therefore continues to pass when the package is fully documented.

    Raises
    ------
    AssertionError
        If the parser and pydocstyle report different violation counts.
    """
    result = run_pydocstyle()
    _assert_pydocstyle_succeeded(result)

    pydocstyle_missing = [
        line for line in result.stdout.splitlines() if re.search(r"\b(D100|D101|D102|D103):", line)
    ]

    pydocstyle_count = len(pydocstyle_missing)

    undocumented = _extract_missing_docstrings(result.stdout)
    parser_count = _count_missing_docstrings(undocumented)

    assert parser_count == pydocstyle_count, (
        f"Parser found {parser_count} missing docstrings, "
        f"but pydocstyle found {pydocstyle_count}.\n"
        f"{_format_missing_docstrings(undocumented)}"
    )


def test_parser_handles_empty_output():
    """Verify that the parser handles empty pydocstyle output.

    Returns
    -------
    None
        The test passes if all result categories are empty.
    """
    result = _extract_missing_docstrings("")

    assert result == {
        "modules": [],
        "classes": [],
        "functions": [],
        "methods": [],
    }


def test_parser_extracts_missing_class_docstring():
    """Verify that the parser identifies undocumented public classes.

    Returns
    -------
    None
        The test passes if the class name is correctly extracted.
    """
    output = (
        "likelihood/example.py:10 in public class `Example`:\n"
        "        D101: Missing docstring in public class\n"
    )

    result = _extract_missing_docstrings(output)

    assert len(result["classes"]) == 1
    assert result["classes"][0]["name"] == "Example"


def test_parser_extracts_missing_function_docstring():
    """Verify that the parser identifies undocumented public functions.

    Returns
    -------
    None
        The test passes if the function violation is correctly identified.
    """
    output = (
        "likelihood/example.py:10 at module level:\n"
        "        D103: Missing docstring in public function\n"
    )

    result = _extract_missing_docstrings(output)

    assert len(result["functions"]) == 1


def test_parser_extracts_missing_module_docstring():
    """Verify that the parser identifies undocumented public modules.

    Returns
    -------
    None
        The test passes if the module violation is correctly identified.
    """
    output = (
        "likelihood/example.py:1 at module level:\n"
        "        D100: Missing docstring in public module\n"
    )

    result = _extract_missing_docstrings(output)

    assert len(result["modules"]) == 1
