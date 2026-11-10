"""Process pydocstyle violations from errores_d401.txt.

Supported rules:
    D205: Require a blank line between summary and description.
    D209: Put closing triple quotes on a separate line.
    D400: Require a period at the end of the summary.
    D401: Require the first line to use imperative mood.
    D105: Report missing docstrings in magic methods.

Usage:
    python fix_docstrings.py --dry-run
    python fix_docstrings.py --dry-run --debug
    python fix_docstrings.py
    python fix_docstrings.py --remove

Backup management:
    cp src/example.py.bak src/example.py
    find . -type f -name "*.bak" -delete
    del /s /q *.bak
"""

import ast
import io
import json
import re
import shutil
import sys
import tokenize
from pathlib import Path

ERRORS_FILE = Path("errores_d401.txt")

SUPPORTED_RULES = {"D205", "D209", "D400", "D401", "D105"}


MANUAL_VERBS = {
    "executes": "Execute",
    "initializes": "Initialize",
    "creates": "Create",
    "transforms": "Transform",
    "removes": "Remove",
    "retrieves": "Retrieve",
    "computes": "Compute",
    "generates": "Generate",
    "calculates": "Calculate",
    "returns": "Return",
    "provides": "Provide",
    "loads": "Load",
    "saves": "Save",
    "reads": "Read",
    "writes": "Write",
    "checks": "Check",
    "determines": "Determine",
    "extracts": "Extract",
    "processes": "Process",
    "performs": "Perform",
    "applies": "Apply",
    "builds": "Build",
    "updates": "Update",
    "finds": "Find",
    "uses": "Use",
    "handles": "Handle",
    "displays": "Display",
    "normalizes": "Normalize",
    "filters": "Filter",
    "sorts": "Sort",
    "validates": "Validate",
    "plots": "Plot",
    "calls": "Call",
    "samples": "Sample",
    "resets": "Reset",
    "takes": "Take",
    "prints": "Print",
    "allows": "Allow",
    "encodes": "Encode",
    "decodes": "Decode",
    "opens": "Open",
    "commits": "Commit",
    "lists": "List",
    "suppresses": "Suppress",
    "decorates": "Decorate",
    "converts": "Convert",
    "returns": "Return",
    "shows": "Show",
    "sets": "Set",
    "performs": "Perform",
    "checks": "Check",
    "contains": "Contain",
    "calculates": "Calculate",
    "determines": "Determine",
}

IRREGULAR_VERBS = {
    "does": "Do",
    "did": "Do",
    "goes": "Go",
    "went": "Go",
    "makes": "Make",
    "made": "Make",
    "takes": "Take",
    "taken": "Take",
    "gets": "Get",
    "gives": "Give",
    "given": "Give",
    "keeps": "Keep",
    "kept": "Keep",
    "sets": "Set",
    "shows": "Show",
    "knows": "Know",
    "known": "Know",
}


PHRASE_FIXES = [
    # General helper descriptions
    (r"^setup\b", "Set up"),
    (r"^internal helper to\b", ""),
    (r"^helper to\b", ""),
    (r"^function that\b", ""),
    (r"^method to\b", ""),
    (r"^method that\b", ""),
    (r"^function to\b", ""),
    # Class and function descriptions
    (r"^the class initializer\b", "Initialize the class"),
    (r"^the initializer of the class\b", "Initialize the class"),
    (r"^the sigmoid function\b", "Compute the sigmoid function"),
    (r"^decorator that\b", "Create a decorator that"),
    (r"^the constructor signature\b", "Return the constructor signature"),
    (r"^the ipywidgets constructor\b", "Return the ipywidgets constructor"),
    (r"^the ``__init__`` of\b", "Return the ``__init__`` of"),
    # Boolean descriptions
    (r"^true when\b", "Check whether"),
    (r"^true for\b", "Check whether"),
    (r"^true if\b", "Check whether"),
    # Names and values
    (r"^simple names of\b", "Return simple names of"),
    (r"^a syntactically valid\b", "Create a syntactically valid"),
    (r"^a value expression\b", "Return a value expression"),
    (r"^the same counters\b", "Return the same counters"),
    (r"^the final meaningful line\b", "Return the final meaningful line"),
    (r"^the json written\b", "Return the JSON written"),
    (r"^the catalog entry\b", "Retrieve the catalog entry"),
    (r"^the open session\b", "Retrieve the open session"),
    (r"^the job with\b", "Retrieve the job with"),
    (r"^the complete\b", "Return the complete"),
    (r"^constructor values\b", "Return constructor values"),
    (r"^required constructor parameters\b", "Return required constructor parameters"),
    # Notebook descriptions
    (r"^a markdown cell\b", "Normalize a markdown cell"),
    (r"^a code cell\b", "Check whether a code cell"),
    (r"^opening cell:", "Create an opening cell:"),
    (r"^the preview template\b", "Load the preview template"),
    # API and server descriptions
    (r"^the request data\b", "Return the request data"),
    (r"^the query string\b", "Return the query string"),
    (r"^the request counters\b", "Return the request counters"),
    (r"^the methods a client may call\b", "Return the methods a client may call"),
    (r"^the most recent jobs\b", "Retrieve the most recent jobs"),
    (r"^dummy arguments\b", "Generate dummy arguments"),
    (r"^the interactive catalogue\b", "Load the interactive catalogue"),
    (r"^a preview without\b", "Handle a preview without"),
    (r"^authentication,", "Configure authentication,"),
    (r"^common headers,", "Return common headers,"),
    (r"^configuration shared by\b", "Configure settings shared by"),
    # Other descriptions
    (r"^a short observation sequence\b", "Return a short observation sequence"),
    (r"^a running fastapi application\b", "Create a running FastAPI application"),
    (r"^the mutable runtime state\b", "Return the mutable runtime state"),
    (r"^deprecated alias\b", "Provide a deprecated alias"),
    (r"^method entries listed in\b", "Return method entries listed in"),
    (r"^resolvable ancestors of\b", "Return resolvable ancestors of"),
    (r"^commits that touched\b", "Return commits that touched"),
    (r"^validated parameter values\b", "Return validated parameter values"),
    (r"^the constructor values\b", "Return constructor values"),
    (r"^the json written after\b", "Return the JSON written after"),
    (r"^the catalog entry of\b", "Retrieve the catalog entry of"),
    (r"^the complete ``atributos`` section\b", "Return the complete ``Atributos`` section"),
]


def read_entries():
    """Read violation entries from the error report."""
    if not ERRORS_FILE.exists():
        raise FileNotFoundError(ERRORS_FILE)

    return ERRORS_FILE.read_text(encoding="utf-8").splitlines()


def parse_entry(entry):
    """Parse a violation and return its file, position, rule and message."""
    notebook = re.match(
        r"^(.*?\.ipynb):cell\s+(\d+):(\d+):(\d+):\s*(D\d+)\s+(.*)$",
        entry,
    )

    if notebook:
        return (
            notebook.group(1),
            int(notebook.group(3)),
            int(notebook.group(4)),
            notebook.group(5),
            notebook.group(6),
            int(notebook.group(2)),
        )

    regular = re.match(
        r"^(.*?):(\d+):(\d+):\s*(D\d+)\s+(.*)$",
        entry,
    )

    if regular:
        return (
            regular.group(1),
            int(regular.group(2)),
            int(regular.group(3)),
            regular.group(4),
            regular.group(5),
            None,
        )

    raise ValueError(f"Invalid entry: {entry}")


def extract_d401_suggestion(message):
    """Extract the original and suggested words from a D401 message."""
    match = re.search(
        r"\(perhaps\s+['\"]([^'\"]+)['\"],?\s+not\s+['\"]([^'\"]+)['\"]\)",
        message,
        re.IGNORECASE,
    )

    if match:
        return match.group(2), match.group(1)

    return None, None


def make_backup(path):
    """Create a backup before modifying a file."""
    backup = path.with_suffix(path.suffix + ".bak")

    if not backup.exists():
        shutil.copy2(path, backup)

    return backup


def read_python_source(path):
    """Read Python source using its declared encoding."""
    with path.open("rb") as file:
        encoding, _ = tokenize.detect_encoding(file.readline)

    return path.read_bytes().decode(encoding), encoding


def find_docstring_node(source, lineno):
    """Find the AST node containing a docstring at the reported line."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue

        body = getattr(node, "body", [])

        if not body:
            continue

        first = body[0]

        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
            and first.lineno <= lineno <= first.end_lineno
        ):
            return node

    return None


def find_string_token(source, node):
    """Find the original string token corresponding to an AST docstring."""
    body = getattr(node, "body", [])

    if not body:
        return None

    first = body[0]

    if not (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return None

    target = first.value

    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.STRING and token.start == (target.lineno, target.col_offset):
                return token

    except (tokenize.TokenError, IndentationError):
        return None

    return None


def get_offset(source, position):
    """Convert a token line and column into a source offset."""
    lines = source.splitlines(keepends=True)
    line, column = position

    return sum(len(item) for item in lines[: line - 1]) + column


def compute_d401(docstring, suggested_word=None):
    """Convert descriptive docstring summaries into imperative mood."""
    text = docstring.strip()

    if not text:
        return None, "empty"

    first_line = text.splitlines()[0].strip()

    # First: known phrase corrections.
    for pattern, replacement in PHRASE_FIXES:
        match = re.match(pattern, first_line, re.IGNORECASE)

        if match:
            remainder = first_line[match.end() :].lstrip()

            if replacement:
                result = replacement

                if remainder:
                    result += " " + remainder
            else:
                if not remainder:
                    return None, "empty_phrase"

                result = remainder[0].upper() + remainder[1:]

            return result, "phrase"

    # Second: pydocstyle suggestion.
    first_word = re.match(r"[A-Za-z]+", first_line)

    if not first_word:
        return None, "unrecognized"

    if suggested_word:
        result = suggested_word.capitalize() + first_line[first_word.end() :]
        return result, "pydocstyle"

    # Third: known imperative verbs.
    word = first_word.group(0).lower()

    if word in MANUAL_VERBS:
        result = MANUAL_VERBS[word] + first_line[first_word.end() :]
        return result, "verb"

    if word in IRREGULAR_VERBS:
        result = IRREGULAR_VERBS[word] + first_line[first_word.end() :]
        return result, "irregular"

    return None, f"requires manual review: {word}"


def transform_docstring_token(token_text, rule, replacement=None):
    """Modify a docstring token while preserving its quote style."""
    match = re.match(r"(?i)^([rub]*)(\"\"\"|''')", token_text)

    if not match:
        return None

    prefix = match.group(1)
    quote = match.group(2)

    if not token_text.endswith(quote):
        return None

    body = token_text[match.end() : -len(quote)]

    newline = "\r\n" if "\r\n" in token_text else "\n"
    lines = body.splitlines(keepends=True)

    if not lines:
        return None

    if rule == "D401":
        if replacement is None:
            return None

        for i, line in enumerate(lines):
            if line.strip():
                leading = line[: len(line) - len(line.lstrip())]

                ending = (
                    "\r\n"
                    if line.endswith("\r\n")
                    else "\n" if line.endswith("\n") else "\r" if line.endswith("\r") else ""
                )

                lines[i] = leading + replacement.rstrip() + ending
                break
        else:
            return None

    elif rule == "D400":
        for i, line in enumerate(lines):
            content = line.rstrip("\r\n")

            if content.strip():
                stripped = content.rstrip()

                if stripped[-1] not in ".!?":
                    lines[i] = stripped + "." + content[len(stripped) :] + line[len(content) :]

                break

    elif rule == "D205":
        summary_index = next(
            (i for i, line in enumerate(lines) if line.strip()),
            None,
        )

        if summary_index is None:
            return None

        next_index = summary_index + 1

        if next_index < len(lines) and lines[next_index].strip():
            lines.insert(next_index, newline)

    elif rule == "D209":
        if len(body.splitlines()) < 2:
            return None

        if not body.endswith(("\n", "\r")):
            lines.append(newline)

    else:
        return None

    updated = prefix + quote + "".join(lines) + quote

    return updated if updated != token_text else None


def fix_source(source, lineno, rule, replacement=None):
    """Apply a docstring correction to the original source."""
    node = find_docstring_node(source, lineno)

    if node is None:
        return None

    token = find_string_token(source, node)

    if token is None:
        return None

    updated = transform_docstring_token(
        token.string,
        rule,
        replacement,
    )

    if updated is None:
        return None

    start = get_offset(source, token.start)
    end = get_offset(source, token.end)

    return source[:start] + updated + source[end:]


def apply_python_fix(file, lineno, rule, replacement=None):
    """Apply a correction to a Python source file."""
    path = Path(file)

    try:
        source, encoding = read_python_source(path)
    except (OSError, SyntaxError, UnicodeError) as exc:
        print(f"    ERROR reading file: {exc}")
        return False

    updated = fix_source(source, lineno, rule, replacement)

    if updated is None or updated == source:
        return False

    make_backup(path)
    path.write_bytes(updated.encode(encoding))

    print(f"    UPDATED: {file}:{lineno} ({rule})")

    return True


def apply_notebook_fix(file, cell_index, lineno, rule, replacement=None):
    """Apply a correction to a specific Jupyter notebook cell."""
    path = Path(file)

    try:
        raw = path.read_bytes()
        notebook = json.loads(raw.decode("utf-8"))

    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"    ERROR reading notebook: {exc}")
        return False

    cells = notebook.get("cells", [])

    if not 0 <= cell_index < len(cells):
        return False

    cell = cells[cell_index]

    if cell.get("cell_type") != "code":
        return False

    original = cell.get("source", [])

    source = "".join(original) if isinstance(original, list) else original

    updated = fix_source(source, lineno, rule, replacement)

    if updated is None or updated == source:
        return False

    cell["source"] = updated.splitlines(keepends=True) if isinstance(original, list) else updated

    make_backup(path)

    output = json.dumps(
        notebook,
        indent=2,
        ensure_ascii=False,
    )

    if b"\r\n" in raw:
        output = output.replace("\r\n", "\n").replace("\n", "\r\n")

    path.write_bytes((output + "\n").encode("utf-8"))

    print(f"    UPDATED: {file}:cell {cell_index} ({rule})")

    return True


def apply_fix(file, lineno, notebook_cell, rule, replacement=None):
    """Dispatch corrections to Python files or notebook cells."""
    if rule == "D105":
        print(f"    MANUAL REVIEW: add a docstring at {file}:{lineno}")
        return False

    if notebook_cell is not None:
        return apply_notebook_fix(
            file,
            notebook_cell,
            lineno,
            rule,
            replacement,
        )

    return apply_python_fix(
        file,
        lineno,
        rule,
        replacement,
    )


def remove_entries(entries, processed):
    """Remove successfully processed violations from the report."""
    remove_set = set(processed)

    kept = [
        line
        for line in entries
        if (not line.strip() or line.strip().startswith("Found") or line.strip() not in remove_set)
    ]

    ERRORS_FILE.write_text(
        "\n".join(kept) + "\n",
        encoding="utf-8",
    )

    print(f"Removed {len(remove_set)} entries from {ERRORS_FILE.name}.")


def main():
    """Process supported pydocstyle violations."""
    dry_run = "--dry-run" in sys.argv
    debug = "--debug" in sys.argv
    do_remove = "--remove" in sys.argv

    entries = read_entries()

    valid_entries = [
        entry for entry in entries if entry.strip() and not entry.strip().startswith("Found")
    ]

    print(f"Total entries: {len(valid_entries)}")

    fixed = 0
    would_fix = 0
    skipped = []
    processed = []

    counts = {rule: 0 for rule in SUPPORTED_RULES}
    reasons = {}

    def skip(entry, reason):
        skipped.append(entry)
        reasons[reason] = reasons.get(reason, 0) + 1

    for i, entry in enumerate(valid_entries):
        try:
            (
                file,
                lineno,
                col,
                rule,
                message,
                notebook_cell,
            ) = parse_entry(entry)

        except ValueError as exc:
            print(f"[{i}] PARSE ERROR: {exc}")
            skip(entry, "parse_error")
            continue

        if rule not in SUPPORTED_RULES:
            print(f"[{i}] UNSUPPORTED: {rule}")
            skip(entry, "unsupported")
            continue

        if not Path(file).exists():
            print(f"[{i}] FILE MISSING: {file}")
            skip(entry, "missing_file")
            continue

        replacement = None

        if rule == "D401":
            _, suggested_word = extract_d401_suggestion(message)

            try:
                if notebook_cell is not None:
                    notebook = json.loads(Path(file).read_text(encoding="utf-8"))

                    cell_source = notebook["cells"][notebook_cell]["source"]

                    source = "".join(cell_source) if isinstance(cell_source, list) else cell_source
                else:
                    source, _ = read_python_source(Path(file))

                node = find_docstring_node(source, lineno)

                if node is None:
                    raise ValueError("Docstring not found")

                actual_docstring = ast.get_docstring(
                    node,
                    clean=False,
                )

            except (
                OSError,
                ValueError,
                KeyError,
                IndexError,
                UnicodeError,
                json.JSONDecodeError,
            ) as exc:
                print(f"[{i}] D401 SOURCE ERROR: {exc}")
                skip(entry, "source_error")
                continue

            replacement, status = compute_d401(
                actual_docstring,
                suggested_word,
            )

            if debug:
                print(f"    Suggested word: {suggested_word!r}")
                print(f"    Replacement: {replacement!r}")
                print(f"    Decision: {status}")

            if replacement is None:
                print(f"[{i}] D401 MANUAL REVIEW: " f"{actual_docstring.splitlines()[0]!r}")
                print(f"    Reason: {status}")
                skip(entry, "manual_review")
                continue

            print(f"[{i}] D401 FIX: " f"{actual_docstring.splitlines()[0]} -> {replacement}")

        elif rule == "D105":
            print(f"[{i}] D105 MANUAL REVIEW: {file}:{lineno}")
            skip(entry, "manual_review")
            continue

        else:
            print(f"[{i}] {rule}: {file}:{lineno}")

        if debug:
            print(f"    Message: {message}")

        would_fix += 1

        if dry_run:
            continue

        try:
            success = apply_fix(
                file,
                lineno,
                notebook_cell,
                rule,
                replacement,
            )

            if success:
                fixed += 1
                counts[rule] += 1
                processed.append(entry)

            else:
                print("    SKIPPED: no safe modification possible")
                skip(entry, "apply_error")

        except Exception as exc:
            print(f"    ERROR: {exc}")
            skip(entry, "apply_error")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    if dry_run:
        print("Dry run: no files were modified.")

    print(f"Fixed:     {fixed}")
    print("Already:   0")
    print(f"Skipped:   {len(skipped)}")

    if dry_run:
        print(f"Would fix: {would_fix}")

    print("\nFIXES BY RULE")

    for rule, count in sorted(counts.items()):
        print(f"{rule}: {count}")

    if reasons:
        print("\nSKIP REASONS")

        for reason, count in sorted(reasons.items()):
            print(f"{reason:18}: {count}")

    if do_remove and not dry_run:
        remove_entries(entries, processed)


if __name__ == "__main__":
    main()
