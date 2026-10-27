#!/usr/bin/env python3
"""Single source of truth for the models shipped with the ``likelihood`` package.

The catalog is built by parsing the package sources with :mod:`ast`.  Nothing is
imported, so the catalog can be generated on any machine -- even when ``numpy``,
``tensorflow`` or ``torch`` are absent -- and there is no hand-maintained JSON
file to drift out of sync with the code.

Only the *layout* of the package is assumed:

* every ``*.py`` module directly under ``likelihood/models`` is a category whose
  name is the module stem (``regression``, ``hmm``, ``ensemble``, ...);
* every non-private module inside a sub-package (``likelihood/models/deep/*.py``)
  belongs to the sub-package category (``deep``).

Everything else -- which classes are models, their parameters, their methods,
their documentation and their source -- is discovered from the code itself.

Usage
-----
>>> from model_catalog import build_catalog
>>> catalog = build_catalog()
>>> catalog[0]["id"], catalog[0]["category"]
('Arima', 'regression')

CLI::

    python3 model_catalog.py --list
    python3 model_catalog.py --json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = REPO_ROOT / "likelihood" / "models"

# Presentation-only metadata.  Categories themselves are discovered from the
# package layout; anything unknown falls back to ``DEFAULT_CATEGORY``.
CATEGORY_META: Dict[str, Tuple[str, str]] = {
    "regression": ("Regression", "📈"),
    "deep": ("Deep Learning", "🧠"),
    "hmm": ("Hidden Markov Models", "🔮"),
    "ensemble": ("Ensemble", "🔗"),
    "simulation": ("Simulation", "🎯"),
    "environments": ("Reinforcement Learning", "🤖"),
    "utils": ("Utilities", "🧰"),
}
DEFAULT_CATEGORY: Tuple[str, str] = ("Models", "📦")

_SECTION_UNDERLINE = re.compile(r"^-{3,}\s*$")
_SECTION_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9 _/-]*$")
_CAMEL_BOUNDARY = re.compile(r"(.)([A-Z][a-z]+)")
_ACRONYM_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_DOTTED = re.compile(r"\s*\.\s*")


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------


def split_identifier(name: str) -> str:
    """Turn a CamelCase identifier into a human readable title.

    ``FourierRegression`` -> ``Fourier Regression``, ``HMM`` -> ``HMM``.
    """
    spaced = _CAMEL_BOUNDARY.sub(r"\1 \2", name)
    spaced = _ACRONYM_BOUNDARY.sub(r"\1 \2", spaced)
    return spaced.strip()


def category_meta(category: str) -> Tuple[str, str]:
    """Return the ``(label, icon)`` pair used to present ``category``."""
    return CATEGORY_META.get(category, DEFAULT_CATEGORY)


def dedent_lines(text: str) -> List[str]:
    """``textwrap.dedent``-like helper that tolerates blank first lines."""
    lines = text.expandtabs().splitlines()
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    if not indents:
        return []
    common = min(indents)
    return [line[common:] if len(line) >= common else line for line in lines]


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _unparse(node: Optional[ast.AST]) -> Optional[str]:
    """Best-effort source text for a node, independent of the Python version."""
    if node is None:
        return None
    if hasattr(ast, "unparse"):
        try:
            return ast.unparse(node)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive
            pass
    return None


def _annotation(node: Optional[ast.AST], fallback: str = "Any") -> str:
    """Render a type annotation as a compact string."""
    text = _unparse(node)
    if not text:
        return fallback
    return _DOTTED.sub(".", text.strip())


def _default(node: Optional[ast.AST]) -> Optional[str]:
    """Render a default value as source, or ``None`` when absent."""
    if node is None:
        return None
    return _unparse(node)


def _decorators(node: ast.AST) -> List[str]:
    """Simple names of the decorators applied to ``node``."""
    names: List[str] = []
    for decorator in getattr(node, "decorator_list", []):
        text = _unparse(decorator) or ""
        names.append(text.split("(")[0].strip().split(".")[-1])
    return names


def _method_kind(decorators: Sequence[str]) -> str:
    """Classify a method by its decorators: ``property`` / ``static`` / ``method``."""
    if "property" in decorators:
        return "property"
    if "staticmethod" in decorators:
        return "static"
    if "classmethod" in decorators:
        return "class"
    return "method"


def guess_widget(type_hint: str) -> str:
    """Map a type annotation onto a UI/ipywidgets control kind.

    Only the *outer* type counts: ``Dict[int, Dict[str, List]]`` is a mapping,
    not an integer, so it must not be rendered as a number input.
    """
    base = re.split(r"[\[\|,]", type_hint)[0].strip().strip("`").strip()
    base = base.split(".")[-1].lower()
    if base in ("int", "float", "bool", "str"):
        return base
    if base == "path" or base.endswith("path"):
        return "text"
    return "raw"


# ---------------------------------------------------------------------------
# Callable introspection
# ---------------------------------------------------------------------------


def _argument_nodes(args: ast.arguments) -> List[Tuple[ast.arg, str, Optional[ast.AST]]]:
    """Yield ``(node, kind, default)`` for every declared argument."""
    positional = list(args.posonlyargs) + list(args.args)
    padding: List[Optional[ast.AST]] = [None] * (len(positional) - len(args.defaults))
    defaults: List[Optional[ast.AST]] = padding + list(args.defaults)

    collected: List[Tuple[ast.arg, str, Optional[ast.AST]]] = []
    for node, default in zip(positional, defaults):
        collected.append((node, "positional", default))
    if args.vararg is not None:
        collected.append((args.vararg, "var_positional", None))
    for node, default in zip(args.kwonlyargs, args.kw_defaults):
        collected.append((node, "keyword_only", default))
    if args.kwarg is not None:
        collected.append((args.kwarg, "var_keyword", None))
    return collected


def extract_parameters(
    func: ast.FunctionDef,
    descriptions: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Describe every argument of ``func`` using annotations + docstring text."""
    descriptions = descriptions or {}
    parameters: List[Dict[str, Any]] = []
    skip = {"self", "cls"}
    for node, kind, default in _argument_nodes(func.args):
        name = node.arg
        if name in skip:
            continue
        type_hint = _annotation(node.annotation)
        documented = descriptions.get(name, "")
        default_text = _default(default)
        parameters.append(
            {
                "name": name,
                "type": type_hint,
                "default": default_text,
                "required": default_text is None and kind in ("positional", "keyword_only"),
                "kind": kind,
                "variadic": kind in ("var_positional", "var_keyword"),
                "description": documented,
                "widget": guess_widget(type_hint),
            }
        )
    return parameters


def _signature_text(func: ast.FunctionDef, name: Optional[str] = None) -> str:
    """Rebuild a compact ``name(a, b=1) -> Ret`` signature from the AST."""
    parts: List[str] = []
    for parameter in extract_parameters(func):
        prefix = ""
        if parameter["kind"] == "var_positional":
            prefix = "*"
        elif parameter["kind"] == "var_keyword":
            prefix = "**"
        text = f"{prefix}{parameter['name']}"
        if not parameter["variadic"] and parameter["type"] != "Any":
            text += f": {parameter['type']}"
        if parameter["default"] is not None:
            text += f" = {parameter['default']}"
        parts.append(text)
    returns = _annotation(func.returns, fallback="")
    suffix = f" -> {returns}" if returns and returns != "None" else ""
    return f"{name or func.name}({', '.join(parts)}){suffix}"


def split_sections(docstring: str) -> Dict[str, List[str]]:
    """Split a numpydoc docstring into ``{section name: lines}``.

    The prose that introduces the docstring is stored under the ``""`` key.
    Sections are detected as ``Title`` followed by a line of dashes.
    """
    lines = dedent_lines(docstring)
    sections: Dict[str, List[str]] = {"": []}
    current = ""
    index = 0
    while index < len(lines):
        line = lines[index]
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if (
            line.strip()
            and _SECTION_UNDERLINE.match(following)
            and _SECTION_NAME.match(line.strip())
        ):
            current = line.strip()
            sections.setdefault(current, [])
            index += 2
            continue
        sections[current].append(line)
        index += 1
    return sections


def _first_paragraph(lines: Sequence[str]) -> str:
    """Return the first non-empty paragraph of ``lines`` joined by spaces."""
    paragraph: List[str] = []
    started = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if started:
                break
            continue
        started = True
        paragraph.append(stripped)
    return " ".join(paragraph)


def _join_description(lines: Sequence[str]) -> str:
    """Collapse a block of indented description lines into one paragraph."""
    return " ".join(line.strip() for line in lines if line.strip())


def _looks_like_entry(name: str) -> bool:
    """True when a docstring line start looks like a ``name`` entry."""
    return bool(name) and " " not in name.replace(",", "")


def parse_typed_entries(lines: Sequence[str]) -> List[Dict[str, str]]:
    """Parse ``name : type`` entries from a docstring section.

    Works for ``Parameters``, ``Attributes`` and ``Other Parameters``.  Text
    lines that belong to the previous entry become its ``description``.
    """
    entries: List[Dict[str, Any]] = []
    for raw in lines:
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip())
        stripped = raw.strip()
        is_new_entry = not entries or indent <= entries[-1]["indent"]
        if is_new_entry and ":" in stripped:
            name, _, type_hint = stripped.partition(":")
            name = name.strip().strip("`")
            if _looks_like_entry(name):
                entries.append(
                    {"name": name, "type": type_hint.strip(), "indent": indent, "desc": []}
                )
                continue
        if entries:
            entries[-1]["desc"].append(stripped)
    return [
        {
            "name": entry["name"],
            "type": entry["type"],
            "description": _join_description(entry["desc"]),
        }
        for entry in entries
    ]


def parse_labelled_entries(lines: Sequence[str]) -> List[Dict[str, str]]:
    """Parse ``label`` + indented description blocks (used by ``Methods``)."""
    entries: List[Dict[str, Any]] = []
    for raw in lines:
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip())
        stripped = raw.strip()
        if not entries or indent <= entries[-1]["indent"]:
            entries.append({"name": stripped, "indent": indent, "desc": []})
        else:
            entries[-1]["desc"].append(stripped)
    return [
        {"name": entry["name"], "description": _join_description(entry["desc"])}
        for entry in entries
    ]


# ---------------------------------------------------------------------------
# Model extraction
# ---------------------------------------------------------------------------


_SAMPLE_LITERALS = {
    "int": "1",
    "float": "1.0",
    "bool": "False",
    "str": '"example"',
    "path": '"model"',
}

# Neutral, runnable placeholders for required string arguments whose names are
# conventional in machine learning.  They are presentation defaults only: the
# user is expected to edit them, and anything unknown falls back to "example".
STRING_PLACEHOLDERS = {
    "activation": "relu",
    "classifier_activation": "softmax",
    "hidden_activation": "relu",
    "output_activation": "linear",
    "optimizer": "adam",
    "loss": "mse",
    "metric": "accuracy",
    "voting_method": "soft",
    "method": "soft",
    "mode": "train",
    "name": "example",
    "filename": "model",
    "path": "model",
    "device": "cpu",
}


@lru_cache(maxsize=None)
def _source_lines(path: str) -> Tuple[str, ...]:
    """Read a source file once and cache its lines."""
    with open(path, "r", encoding="utf-8") as handle:
        return tuple(handle.read().splitlines(keepends=True))


def _node_source(path: Path, node: ast.AST) -> str:
    """Return the exact source text that produced ``node``."""
    lines = _source_lines(str(path))
    start = getattr(node, "lineno", 1) - 1
    end = getattr(node, "end_lineno", None) or len(lines)
    return "".join(lines[start:end]).rstrip("\n")


def _callable_records(cls: ast.ClassDef) -> List[ast.FunctionDef]:
    """Every method directly defined by ``cls``."""
    return [node for node in cls.body if isinstance(node, ast.FunctionDef)]


def _public_methods(cls: ast.ClassDef) -> List[ast.FunctionDef]:
    """Methods that are part of the public surface of ``cls``."""
    return [
        method
        for method in _callable_records(cls)
        if not method.name.startswith("_") and method.name != "__init__"
    ]


def _init_function(cls: ast.ClassDef) -> Optional[ast.FunctionDef]:
    """The ``__init__`` of ``cls`` when it declares one."""
    for method in _callable_records(cls):
        if method.name == "__init__":
            return method
    return None


def _documented_methods(docstring: str) -> List[Dict[str, str]]:
    """Method entries listed in the ``Methods`` section of ``docstring``."""
    sections = split_sections(docstring or "")
    lines = sections.get("Methods") or sections.get("Public Methods") or []
    return parse_labelled_entries(lines)


def _method_index(cls: ast.ClassDef, docstring: str) -> List[Dict[str, Any]]:
    """Merge the real methods of ``cls`` with what its docstring documents."""
    documented = {
        entry["name"].split("(")[0].strip(): entry["description"]
        for entry in _documented_methods(docstring)
    }
    methods: List[Dict[str, Any]] = []
    for func in _public_methods(cls):
        description = documented.pop(func.name, "")
        if not description:
            own = split_sections(ast.get_docstring(func) or "")
            description = _first_paragraph(own.get("", []))
        decorators = _decorators(func)
        methods.append(
            {
                "name": func.name,
                "signature": _signature_text(func),
                "description": description,
                "parameters": extract_parameters(func),
                "kind": _method_kind(decorators),
                "decorators": decorators,
                "inherited": False,
            }
        )
    # Methods that only exist in the documentation (inherited from a parent).
    for name, description in documented.items():
        methods.append(
            {
                "name": name.split("(")[0].strip(),
                "signature": name,
                "description": description,
                "parameters": [],
                "kind": "method",
                "decorators": [],
                "inherited": True,
            }
        )
    return methods


_CONTAINER_SAMPLES: Tuple[Tuple[str, str], ...] = (
    ("ndarray", "np.array([])"),
    ("array", "np.array([])"),
    ("dataframe", "pd.DataFrame()"),
    ("series", "pd.Series(dtype=float)"),
    ("dict", "{}"),
    ("mapping", "{}"),
    ("list", "[]"),
    ("tuple", "()"),
    ("set", "set()"),
)


def _sample_expression(parameter: Dict[str, Any]) -> str:
    """A syntactically valid placeholder value for ``parameter``."""
    type_hint = parameter["type"]
    widget = guess_widget(type_hint)
    if widget in ("str", "text"):
        name = parameter["name"].lower().strip("_")
        value = STRING_PLACEHOLDERS.get(name)
        if value is None and name.endswith("_path"):
            value = "model"
        return repr(value or "example")
    if widget in _SAMPLE_LITERALS:
        return _SAMPLE_LITERALS[widget]
    lowered = type_hint.lower()
    for hint, literal in _CONTAINER_SAMPLES:
        if hint in lowered:
            return literal
    return "None"


def _parameter_name(name: str) -> str:
    """Sanitise a parameter name so it can be used as a Python identifier."""
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    return f"p_{cleaned}" if cleaned[:1].isdigit() else (cleaned or "parameter")


_SIDE_EFFECT_METHODS = ("save", "load", "plot", "write", "download", "delete", "reset", "remove")


def _is_literal(value: str) -> bool:
    """True when ``value`` is a scalar literal that can be inlined in a call."""
    if value in ("True", "False", "None"):
        return True
    if value.startswith(('"', "'")):
        return True
    try:
        float(value)
    except ValueError:
        return False
    return True


def _call_safe_methods(model: Dict[str, Any], limit: int = 2) -> List[str]:
    """Public methods that can be called without arguments and without side effects."""
    calls: List[str] = []
    for method in model["methods"]:
        if method["kind"] == "property":
            continue  # accessed as an attribute, never called
        if method["inherited"] and not method["parameters"]:
            continue  # documented only: the signature is unknown
        if any(parameter["required"] for parameter in method["parameters"]):
            continue
        if any(token in method["name"].lower() for token in _SIDE_EFFECT_METHODS):
            continue
        calls.append(f"model.{method['name']}()")
        if len(calls) >= limit:
            break
    return calls


def _example_code(model: Dict[str, Any]) -> str:
    """Generate a usage example straight from the real constructor signature."""
    required = [p for p in model["parameters"] if p["required"] and not p["variadic"]]
    optional = [p for p in model["parameters"] if not p["required"] and not p["variadic"]]

    arguments: List[str] = []
    setup: List[str] = []
    for parameter in required:
        name = _parameter_name(parameter["name"])
        expression = parameter["sample"]
        if _is_literal(expression):
            arguments.append(f"{parameter['name']}={expression}")
            continue
        setup.append(f"{name} = {expression}  # {parameter['type']}")
        arguments.append(f"{parameter['name']}={name}")

    body = "\n".join(setup)
    imports = [model["importStatement"]]
    if "np." in body:
        imports.append("import numpy as np")
    if "pd." in body:
        imports.append("import pandas as pd")

    lines = [*imports, ""]
    if setup:
        lines.extend(setup)
        lines.append("")
    lines.append(f"model = {model['name']}({', '.join(arguments)})")

    calls = model.get("safeCalls") or []
    if calls:
        lines.append("")
        lines.append("# Ejemplo de uso")
        lines.extend(f"# {call}" for call in calls)

    if optional:
        summary = ", ".join(
            f"{p['name']}={p['default']}" if p["default"] is not None else p["name"]
            for p in optional
        )
        lines.append("")
        lines.append(f"# Opcionales: {summary}")
    return "\n".join(lines).strip() + "\n"


def _plural(count: int, singular: str, plural: str) -> str:
    """``_plural(1, "método", "métodos")`` -> ``"1 método"``."""
    return f"{count} {singular if count == 1 else plural}"


def _tags(model: Dict[str, Any]) -> List[str]:
    """Derive search tags from the category, the name and the class hierarchy."""
    tags: List[str] = [model["categoryLabel"]]
    for word in split_identifier(model["name"]).split():
        if len(word) > 2 and word.lower() not in {tag.lower() for tag in tags}:
            tags.append(word)
    for base in model.get("bases", []):
        if base not in tags:
            tags.append(base)
    if model["methods"]:
        tags.append(_plural(len(model["methods"]), "método", "métodos"))
    if model["parameters"]:
        tags.append(_plural(len(model["parameters"]), "parámetro", "parámetros"))
    return tags


def _to_model(record: Dict[str, Any]) -> Dict[str, Any]:
    """Turn an AST record into the public catalog entry."""
    model = dict(record)
    label, icon = category_meta(model["category"])
    model["categoryLabel"] = label
    model["icon"] = icon
    model["shortDescription"] = _first_paragraph(split_sections(model["docstring"]).get("", []))
    model["fullDescription"] = model["docstring"].strip() or model["shortDescription"]
    model["title"] = split_identifier(model["name"])
    model["importStatement"] = f"from {model['module']} import {model['name']}"
    model["parameters"] = [
        {**parameter, "sample": _sample_expression(parameter)} for parameter in model["parameters"]
    ]
    model["safeCalls"] = _call_safe_methods(model)
    model["example"] = _example_code(model)
    model["tags"] = _tags(model)
    model["lineCount"] = model["source"].count("\n") + 1
    for private in ("_node", "docstring"):
        model.pop(private, None)
    return model


def _category_of(module_path: Path, package_dir: Path) -> str:
    """Category name for ``module_path`` derived purely from the layout."""
    relative = module_path.relative_to(package_dir)
    parts = relative.parts
    if len(parts) == 1:
        return module_path.stem if module_path.stem != "__init__" else ""
    # Inside a sub-package: use the sub-package directory name.
    return parts[0]


def _module_name(module_path: Path, package_dir: Path, package: str) -> str:
    """Dotted import path of ``module_path`` (``likelihood.models.deep.gan``)."""
    relative = module_path.relative_to(package_dir).with_suffix("")
    parts = [part for part in relative.parts if part != "__init__"]
    return ".".join([package, *parts])


def _candidate_classes(tree: ast.Module) -> List[ast.ClassDef]:
    """Public, top-level classes defined by ``tree``.

    Helper *functions* are deliberately ignored: the catalog describes the
    models of the library, and those are always classes (loss functions,
    sampling helpers, callbacks, ... are implementation details).
    """
    return [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    ]


def _base_class_names(cls: ast.ClassDef) -> List[str]:
    """Simple names of the classes ``cls`` inherits from."""
    names: List[str] = []
    for base in cls.bases:
        text = _unparse(base) or ""
        if text:
            names.append(text.split("[")[0].split(".")[-1].strip())
    return names


def _init_arguments(cls: ast.ClassDef) -> List[ast.arg]:
    """Argument nodes of the ``__init__`` of ``cls``."""
    init = _init_function(cls)
    return [] if init is None else [arg for arg, _, _ in _argument_nodes(init.args)]


def _collect_module(
    module_path: Path,
    package_dir: Path,
    package: str,
    include_private: bool,
    min_parameters: int,
) -> List[Dict[str, Any]]:
    """Extract every model record defined in a single module."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    category = _category_of(module_path, package_dir)
    module_name = _module_name(module_path, package_dir, package)
    if category == "utils" and not include_private:
        return []

    models: List[Dict[str, Any]] = []
    for node in _candidate_classes(tree):
        if len(_init_arguments(node)) < min_parameters:
            continue
        docstring = ast.get_docstring(node) or ""
        if not docstring and not include_private:
            continue
        sections = split_sections(docstring)
        documented = _documented_parameters(docstring)
        documented.update(
            {
                entry["name"].strip("`"): entry["description"]
                for entry in parse_typed_entries(sections.get("Attributes", []))
            }
        )
        initialiser = _init_function(node)
        assert initialiser is not None
        parameters = extract_parameters(initialiser, documented)
        if any(p["kind"] == "var_keyword" for p in parameters):
            # The class forwards **kwargs, so the documented keyword arguments
            # really can be passed to the constructor.
            parameters.extend(
                _keyword_parameters(docstring, {parameter["name"] for parameter in parameters})
            )
        parameters = [p for p in parameters if p["kind"] != "var_keyword"]
        models.append(
            {
                "id": node.name,
                "name": node.name,
                "kind": "class",
                "category": category,
                "module": module_name,
                "path": str(module_path.relative_to(package_dir.parent.parent)),
                "lineno": getattr(node, "lineno", 0),
                "docstring": docstring,
                "signature": _signature_text(initialiser, name=node.name),
                "parameters": parameters,
                "attributes": parse_typed_entries(sections.get("Attributes", [])),
                "methods": _method_index(node, docstring),
                "source": _node_source(module_path, node),
                "bases": _base_class_names(node),
                "_node": node,
            }
        )
    return models


def iter_modules(package_dir: Path) -> List[Path]:
    """All candidate modules, sorted, skipping private and cache files."""
    modules: List[Path] = []
    for path in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if path.name.startswith("_"):
            continue
        modules.append(path)
    return modules


def _public_module_first(path: Path) -> Tuple[int, str]:
    """Sort key that prefers public modules over private ones."""
    return (1 if path.name.startswith("_") else 0, str(path))


def class_index(package_dir: Path) -> Dict[str, ast.ClassDef]:
    """Map every class defined in the surrounding package to its AST node.

    Base classes are resolved through this index so that a model inherits the
    methods and parameter documentation implemented by its parents
    (``Arima`` -> ``AbstractArima``, ``SimulationEngine`` -> ``FeatureSelection``).
    """
    index: Dict[str, ast.ClassDef] = {}
    root = package_dir.parent
    for path in sorted(root.rglob("*.py"), key=_public_module_first):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name not in index:
                index[node.name] = node
    return index


def base_chain(
    cls: ast.ClassDef, index: Dict[str, ast.ClassDef], limit: int = 10
) -> List[ast.ClassDef]:
    """Resolvable ancestors of ``cls``, nearest first."""
    chain: List[ast.ClassDef] = []
    seen = {cls.name}
    queue = list(_base_class_names(cls))
    while queue and len(chain) < limit:
        name = queue.pop(0)
        if name in seen:
            continue
        seen.add(name)
        parent = index.get(name)
        if parent is None:
            continue
        chain.append(parent)
        queue.extend(_base_class_names(parent))
    return chain


def _documented_parameters(docstring: str) -> Dict[str, str]:
    """``{parameter: description}`` taken from every parameter-like section."""
    sections = split_sections(docstring)
    documented: Dict[str, str] = {}
    for section_name in ("Parameters", "Keyword Arguments", "Other Parameters"):
        for entry in parse_typed_entries(sections.get(section_name, [])):
            documented.setdefault(entry["name"].strip("`"), entry["description"])
    return documented


_DEFAULT_IN_TEXT = re.compile(r"[Dd]efaults?\s*(?:is|to|:|=)?\s*`?([^`.,;)\n]+)`?")


def _keyword_parameters(docstring: str, known: set) -> List[Dict[str, Any]]:
    """Recover the keyword arguments a constructor accepts through ``**kwargs``.

    Many models forward optional settings through ``**kwargs`` and document them
    in a ``Keyword Arguments`` section.  Those entries are real, user facing
    parameters, so they are surfaced alongside the declared signature.
    """
    sections = split_sections(docstring)
    entries: List[Dict[str, str]] = []
    for section_name in ("Keyword Arguments", "Other Parameters"):
        entries.extend(parse_typed_entries(sections.get(section_name, [])))

    parameters: List[Dict[str, Any]] = []
    for entry in entries:
        name = entry["name"].strip("`")
        if not name or name in known or name in ("kwargs", "args", "self"):
            continue
        match = _DEFAULT_IN_TEXT.search(entry["description"])
        default = match.group(1).strip() if match else None
        type_hint = entry["type"].strip().strip("`") or "Any"
        parameters.append(
            {
                "name": name,
                "type": type_hint,
                "default": default,
                "required": False,
                "kind": "keyword_only",
                "variadic": False,
                "description": entry["description"],
                "widget": guess_widget(type_hint),
            }
        )
    return parameters


def _merge_inherited(records: Sequence[Dict[str, Any]], index: Dict[str, ast.ClassDef]) -> None:
    """Add the public methods and parameter docs a model inherits from its parents."""
    for record in records:
        own_methods = {method["name"] for method in record["methods"]}
        for ancestor in base_chain(record["_node"], index):
            docstring = ast.get_docstring(ancestor) or ""
            if record.get("inheritedFrom") is None:
                record["inheritedFrom"] = ancestor.name
            for method in _method_index(ancestor, docstring):
                if method["name"] in own_methods:
                    continue
                method["inherited"] = True
                record["methods"].append(method)
                own_methods.add(method["name"])
            described = _documented_parameters(docstring)
            for parameter in record["parameters"]:
                if not parameter["description"]:
                    parameter["description"] = described.get(parameter["name"], "")


def build_catalog(
    package_dir: Optional[Path] = None,
    min_parameters: int = 0,
    include_private: bool = False,
) -> List[Dict[str, Any]]:
    """Introspect the package and return the list of models.

    Parameters
    ----------
    package_dir :
        Directory holding the model modules.  Defaults to
        ``likelihood/models`` next to this file.
    min_parameters :
        Minimum number of constructor arguments a class must declare to be
        considered a model.  ``0`` keeps every documented class.
    include_private :
        Also include undocumented classes and abstract base classes.

    Returns
    -------
    list of dict
        One entry per model, sorted by category and then by title.

    Notes
    -----
    A public class that serves as a base for another discovered class is
    treated as an abstract implementation detail and skipped (for example
    ``AbstractArima``, whose subclasses ``Arima`` and ``FourierRegression``
    are the actual models).
    """
    package_dir = Path(package_dir or DEFAULT_PACKAGE_DIR)
    if not package_dir.is_dir():
        raise FileNotFoundError(f"Model package directory not found: {package_dir}")
    package = _package_name(package_dir)

    records: List[Dict[str, Any]] = []
    for module_path in iter_modules(package_dir):
        records.extend(
            _collect_module(module_path, package_dir, package, include_private, min_parameters)
        )

    _merge_inherited(records, class_index(package_dir))

    if not include_private:
        base_names = {base for record in records for base in record.get("bases", [])}
        records = [record for record in records if record["id"] not in base_names]

    catalog = [_to_model(record) for record in records]
    catalog.sort(key=lambda item: (item["category"], item["title"].lower()))
    return catalog


def _package_name(package_dir: Path) -> str:
    """Reconstruct the dotted package name from an on-disk directory."""
    parts: List[str] = []
    current = package_dir
    while (current / "__init__.py").exists():
        parts.append(current.name)
        current = current.parent
    return ".".join(reversed(parts)) or package_dir.name


def models_by_category(catalog: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group ``catalog`` into ``{category: [models]}``."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for model in catalog:
        grouped.setdefault(model["category"], []).append(model)
    return grouped


def category_index(catalog: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Lightweight category descriptors used by the web UI."""
    grouped = models_by_category(catalog)
    index: List[Dict[str, Any]] = []
    for name in sorted(grouped):
        label, icon = category_meta(name)
        index.append({"id": name, "label": label, "icon": icon, "count": len(grouped[name])})
    return index


def find_model(catalog: Sequence[Dict[str, Any]], model_id: str) -> Optional[Dict[str, Any]]:
    """Look a model up by id, tolerating case differences."""
    for model in catalog:
        if model["id"] == model_id:
            return model
    lowered = model_id.lower()
    for model in catalog:
        if model["id"].lower() == lowered:
            return model
    return None


def filter_by_category(catalog: Sequence[Dict[str, Any]], category: str) -> List[Dict[str, Any]]:
    """Models belonging to ``category`` (case insensitive)."""
    lowered = category.strip().lower()
    return [model for model in catalog if model["category"].lower() == lowered]


def _haystack(model: Dict[str, Any]) -> str:
    """Every searchable text of ``model`` joined into a single string."""
    parts = [
        model["id"],
        model["title"],
        model["category"],
        model["categoryLabel"],
        model["module"],
        model["shortDescription"],
        model["fullDescription"],
        " ".join(model["tags"]),
        " ".join(method["name"] for method in model["methods"]),
        " ".join(parameter["name"] for parameter in model["parameters"]),
    ]
    return " ".join(parts).lower()


def search_models(catalog: Sequence[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Free text search across names, descriptions, tags, methods and parameters."""
    needle = query.strip().lower()
    if not needle:
        return list(catalog)
    terms = [term for term in re.split(r"\s+", needle) if term]
    matches = []
    for model in catalog:
        haystack = _haystack(model)
        if all(term in haystack for term in terms):
            matches.append(model)
    return matches


def summarize(catalog: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """``{category: count}`` plus a ``total`` entry."""
    summary = {name: len(items) for name, items in models_by_category(catalog).items()}
    summary["total"] = len(catalog)
    return summary


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Introspect the likelihood model catalog")
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--all", action="store_true", help="include undocumented definitions")
    parser.add_argument("--json", action="store_true", help="dump the full catalog as JSON")
    parser.add_argument("--list", action="store_true", help="print a human readable summary")
    arguments = parser.parse_args(argv)

    catalog = build_catalog(arguments.package_dir, include_private=arguments.all)
    if arguments.json:
        print(json.dumps(catalog, indent=2, ensure_ascii=False))
        return 0

    for category, count in sorted(summarize(catalog).items()):
        if category == "total":
            print(f"{count:>4}  TOTAL")
        else:
            label, icon = category_meta(category)
            print(f"{count:>4}  {icon} {label} ({category})")
    if arguments.list:
        print()
        for model in catalog:
            print(f"  {model['icon']} {model['id']:<24} {model['module']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
