"""Generate a report of missing NumPy-style docstrings.

Run with::

    python tests/generate_docstring_report.py

The script runs pydocstyle over the likelihood package and generates
``output.txt`` containing undocumented modules, classes, methods,
and functions, together with suggested docstring templates.
"""

import ast
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = REPO_ROOT / "output.txt"


def run_pydocstyle():
    """Run pydocstyle using the NumPy convention.

    Returns
    -------
    subprocess.CompletedProcess
        Captured pydocstyle output and return code.
    """
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pydocstyle",
            "likelihood",
            "--convention=numpy",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def get_enclosing_class(path, lineno):
    """Find the class containing a given line number.

    Parameters
    ----------
    path : pathlib.Path
        Python source file.
    lineno : int
        Line number of the undocumented method.

    Returns
    -------
    ast.ClassDef or None
        Enclosing class, if found.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None

    candidates = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            end_lineno = getattr(node, "end_lineno", node.lineno)

            if node.lineno <= lineno <= end_lineno:
                candidates.append(node)

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda node: (getattr(node, "end_lineno", node.lineno) - node.lineno),
    )


def extract_violations(output):
    """Extract missing docstrings from pydocstyle output.

    Parameters
    ----------
    output : str
        Captured pydocstyle stdout.

    Returns
    -------
    list of dict
        Missing docstring entries with their file, line, name,
        violation code, and entity type.
    """
    violations = []

    lines = output.splitlines()

    for index, line in enumerate(lines):
        match = re.search(r"\b(D100|D101|D102|D103):", line)

        if not match or index == 0:
            continue

        code = match.group(1)
        header = lines[index - 1]

        location = re.match(r"(.+?):(\d+)", header)

        if not location:
            continue

        path = Path(location.group(1))

        if not path.is_absolute():
            path = REPO_ROOT / path

        lineno = int(location.group(2))

        name_match = re.search(r"`+([^`]+)`+", header)
        name = name_match.group(1) if name_match else ""

        entry = {
            "code": code,
            "name": name,
            "path": path,
            "lineno": lineno,
            "type": "",
            "class": None,
        }

        if code == "D100":
            entry["type"] = "module"

        elif code == "D101":
            entry["type"] = "class"

        elif code == "D103":
            entry["type"] = "function"

        elif code == "D102":
            entry["type"] = "method"

            enclosing_class = get_enclosing_class(path, lineno)

            if enclosing_class:
                entry["class"] = enclosing_class.name

        violations.append(entry)

    return violations


def generate_docstring_template(entry):
    """Generate a suggested NumPy-style docstring template.

    Parameters
    ----------
    entry : dict
        Missing docstring information.

    Returns
    -------
    str
        Suggested docstring template.
    """
    entity_type = entry["type"]
    name = entry["name"]

    if entity_type == "module":
        return '''"""
    TODO: Describe the purpose of this module.
    """'''
    if entity_type == "class":
        return f'''    """
    TODO: Describe the purpose of {name}.

    Parameters
    ----------
    TODO
        Describe the class parameters.

    Attributes
    ----------
    TODO
        Describe the class attributes.
    """'''

    if entity_type == "method":
        return f'''    """
    TODO: Describe what {name} does.

    Parameters
    ----------
    TODO
        Describe the method parameters.

    Returns
    -------
    TODO
        Describe the return value.
    """'''

    return f'''"""
    TODO: Describe what {name} does.

    Parameters
    ----------
    TODO
        Describe the function parameters.

    Returns
    -------
    TODO
        Describe the return value.
    """'''


def generate_report(violations):
    """Generate a readable documentation TODO report.

    Parameters
    ----------
    violations : list of dict
        Missing docstring entries.

    Returns
    -------
    str
        Complete report ready to be written to output.txt.
    """
    if not violations:
        return (
            "All public modules, classes, functions, and methods " "have NumPy-style docstrings.\n"
        )

    files = defaultdict(list)

    for entry in violations:
        files[entry["path"]].append(entry)

    output = [
        "=" * 75,
        "MISSING DOCSTRING REPORT",
        "=" * 75,
        "",
        f"Files with missing docstrings: {len(files)}",
        f"Total missing docstrings: {len(violations)}",
        "",
    ]

    for file_index, path in enumerate(sorted(files), start=1):
        entries = sorted(
            files[path],
            key=lambda entry: entry["lineno"],
        )

        try:
            relative_path = path.relative_to(REPO_ROOT)
        except ValueError:
            relative_path = path

        output.extend(
            [
                "=" * 75,
                f"{file_index}. {relative_path}",
                "=" * 75,
                "",
            ]
        )

        for entry in entries:
            name = entry["name"]

            if entry["type"] == "method" and entry["class"]:
                name = f"{entry['class']}.{name}"

            output.extend(
                [
                    f"[{entry['code']}] {entry['type'].upper()}: {name}",
                    f"Line: {entry['lineno']}",
                    "",
                    "Suggested NumPy-style docstring:",
                    generate_docstring_template(entry),
                    "",
                    "-" * 75,
                    "",
                ]
            )

    output.extend(
        [
            "=" * 75,
            f"TOTAL: {len(violations)} missing docstring(s)",
            "=" * 75,
        ]
    )

    return "\n".join(output)


def main():
    """Run the documentation report generator.

    Returns
    -------
    int
        Exit status: zero on successful report generation.
    """
    print("Running pydocstyle...")

    result = run_pydocstyle()

    # Exit code 1 is expected when pydocstyle finds violations.
    # Other nonzero exit codes may indicate execution problems.
    if result.returncode not in (0, 1):
        print("Error executing pydocstyle:")
        print(result.stderr)
        return 1

    if result.stderr:
        print("pydocstyle stderr:")
        print(result.stderr)

    violations = extract_violations(result.stdout)
    report = generate_report(violations)

    OUTPUT_FILE.write_text(report, encoding="utf-8")

    print()
    print(f"Report generated: {OUTPUT_FILE}")
    print(f"Files affected: {len(set(v['path'] for v in violations))}")
    print(f"Missing docstrings: {len(violations)}")

    if not violations:
        print("All public entities are documented!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
