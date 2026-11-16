#!/usr/bin/env python3
"""Notebook generator for the ``likelihood`` models.

Every notebook is derived from :mod:`model_catalog`, which introspects the real
package sources.  As a consequence the generated documentation, signatures,
methods and examples can never drift away from the implementation:

* the model list comes from the package layout, not from a hand written JSON;
* parameter tables and widgets are built from the real ``__init__`` signature;
* the example code is generated from that same signature and stays executable.

When ``nbformat`` is installed the notebooks are validated against the official
notebook schema before being written; otherwise the plain ``dict`` described by
the schema is used.

Usage
-----
Generate every notebook::

    python3 notebook_generator.py --all

Generate a single model, or a whole category::

    python3 notebook_generator.py --single Arima
    python3 notebook_generator.py --category deep
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from model_catalog import (
    REPO_ROOT,
    build_catalog,
    category_meta,
    find_model,
    models_by_category,
    summarize,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "notebooks"
NOTEBOOK_FORMAT = 4
NOTEBOOK_FORMAT_MINOR = 5
KERNEL_DISPLAY_NAME = "Python 3 (likelihood library)"
_INDENT = 2


def _notebook_metadata(model: Dict[str, Any]) -> Dict[str, Any]:
    """Notebook level metadata (a single ``metadata`` key, as nbformat requires)."""
    major, _, rest = sys.version.partition(".")
    minor = rest.split(".")[0]
    return {
        "kernelspec": {
            "display_name": KERNEL_DISPLAY_NAME,
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": f"{major}.{minor}",
            "mimetype": "text/x-python",
            "file_extension": ".py",
        },
        "likelihood": {
            "model": model["id"],
            "category": model["category"],
            "module": model["module"],
        },
        "widgets": {},
        "colab": {"name": model["title"], "provenance": []},
    }


def _empty_notebook(model: Dict[str, Any]) -> Dict[str, Any]:
    """Notebook skeleton for ``model``."""
    return {
        "cells": [],
        "metadata": _notebook_metadata(model),
        "nbformat": NOTEBOOK_FORMAT,
        "nbformat_minor": NOTEBOOK_FORMAT_MINOR,
    }


def _markdown(source: str, tags: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Normalize a markdown cell normalised to a trailing newline."""
    metadata: Dict[str, Any] = {"tags": list(tags)} if tags else {}
    return {"cell_type": "markdown", "metadata": metadata, "source": source.rstrip() + "\n"}


def _code(source: str, tags: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Check whether a code cell that has never been executed."""
    metadata: Dict[str, Any] = {"tags": list(tags)} if tags else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": source.rstrip() + "\n",
    }


def _escape(value: str) -> str:
    """Escape the characters that are special inside a markdown table."""
    return str(value).replace("|", "\\|")


def _parameter_table(parameters: Sequence[Dict[str, Any]]) -> str:
    """Markdown table describing ``parameters``."""
    if not parameters:
        return "_Este modelo no declara parámetros de entrada._"
    lines = ["| Parámetro | Tipo | Por defecto | Descripción |", "| --- | --- | --- | --- |"]
    for parameter in parameters:
        default = parameter["default"] if parameter["default"] is not None else "—"
        lines.append(
            "| `{name}` | `{type}` | `{default}` | {description} |".format(
                name=_escape(parameter["name"]),
                type=_escape(parameter["type"]),
                default=_escape(default),
                description=_escape(parameter["description"]) or "—",
            )
        )
    return "\n".join(lines)


def _method_table(methods: Sequence[Dict[str, Any]]) -> str:
    """Markdown table describing ``methods``."""
    if not methods:
        return "_Este modelo no expone métodos públicos._"
    lines = ["| Método | Origen | Descripción |", "| --- | --- | --- |"]
    for method in methods:
        origin = "heredado" if method["inherited"] else "propio"
        lines.append(
            "| `{signature}` | {origin} | {description} |".format(
                signature=_escape(method["signature"].split(" -> ")[0]),
                origin=origin,
                description=_escape(method["description"]) or "—",
            )
        )
    return "\n".join(lines)


def _attribute_table(attributes: Sequence[Dict[str, Any]]) -> str:
    """Markdown table describing the documented attributes."""
    if not attributes:
        return ""
    lines = ["| Atributo | Tipo | Descripción |", "| --- | --- | --- |"]
    for attribute in attributes:
        lines.append(
            "| `{name}` | `{type}` | {description} |".format(
                name=_escape(attribute["name"]),
                type=_escape(attribute.get("type") or "—"),
                description=_escape(attribute.get("description")) or "—",
            )
        )
    return "\n".join(lines)


def _title_cell(model: Dict[str, Any]) -> Dict[str, Any]:
    """Create an opening cell: title, provenance and summary."""
    label, icon = category_meta(model["category"])
    source = f"# {model['icon']} {model['title']}\n\n"
    source += f"**Categoría:** {icon} {label} `({model['category']})`  \n"
    source += f"**Módulo:** `{model['module']}`  \n"
    source += f"**Clase:** `{model['id']}`  \n"
    source += f"**Fuente:** `{model['path']}:{model['lineno']}`\n\n"
    if model["short_description"]:
        source += f"> {model['short_description']}\n"
    return _markdown(source, tags=["title"])


def _description_cell(model: Dict[str, Any]) -> Dict[str, Any]:
    """Full prose documentation of the model."""
    return _markdown(f"## 📖 Descripción\n\n{model['full_description']}\n")


def _usage_cell(model: Dict[str, Any]) -> Dict[str, Any]:
    """Usage example generated from the real signature."""
    return _markdown("## ⚡ Inicio rápido\n\n```python\n" + model["example"].rstrip() + "\n```\n")


def _signature_cell(model: Dict[str, Any]) -> Dict[str, Any]:
    """Return the constructor signature and the import statement."""
    source = "## 🧩 Constructor\n\n```python\n"
    source += f"{model['import_statement']}\n\n{model['signature']}\n```\n"
    if model.get("inheritedFrom"):
        source += f"\nHereda de `{model['inheritedFrom']}`.\n"
    return _markdown(source)


def _parameters_cell(model: Dict[str, Any]) -> Dict[str, Any]:
    """Parameter table plus the documented attributes, when available."""
    source = "## 🔧 Parámetros\n\n" + _parameter_table(model["parameters"]) + "\n"
    attributes = _attribute_table(model.get("attributes", []))
    if attributes:
        source += f"\n### Atributos\n\n{attributes}\n"
    return _markdown(source)


def _methods_cell(model: Dict[str, Any]) -> Dict[str, Any]:
    """Table of the public API of the model."""
    return _markdown("## 📋 Métodos disponibles\n\n" + _method_table(model["methods"]) + "\n")


def _is_safe_literal(text: Optional[str]) -> bool:
    """Check whether ``text`` is a literal (tuple, number, quoted string, ``None``)."""
    if not text:
        return False
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:
        return False
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id not in ("True", "False", "None"):
            return False
        if isinstance(child, (ast.Call, ast.Attribute, ast.Subscript)):
            return False
    return True


def _split_parameters(
    model: Dict[str, Any],
) -> tuple:
    """Split the parameters into widget driven ones and fixed ones."""
    tunable: List[Dict[str, Any]] = []
    fixed: List[Dict[str, Any]] = []
    for parameter in model["parameters"]:
        if parameter["variadic"]:
            continue
        if parameter["widget"] in ("int", "float", "bool", "str"):
            tunable.append(parameter)
        else:
            fixed.append(parameter)
    return tunable, fixed


def _widget_value(parameter: Dict[str, Any]) -> str:
    """Return a value expression that is valid for the widget of ``parameter``."""
    kind = parameter["widget"]
    default = parameter["default"]
    if kind == "bool":
        return "True" if default == "True" else "False"
    if kind == "int":
        try:
            return str(int(float(default)))
        except (TypeError, ValueError):
            return "1"
    if kind == "float":
        try:
            return repr(float(default))
        except (TypeError, ValueError):
            return "0.0"
    # Strings: keep the documented default, otherwise reuse the same neutral
    # placeholder that the generated example uses.
    if default is None:
        return parameter["sample"]
    return default if default.startswith(("'", '"')) else repr(default)


def _widget_expression(parameter: Dict[str, Any]) -> str:
    """Return the ipywidgets constructor for ``parameter``."""
    name = parameter["name"]
    value = _widget_value(parameter)
    kind = parameter["widget"]
    if kind == "bool":
        return f"Checkbox(value={value}, description='{name}')"
    if kind == "int":
        return f"IntSlider(value={value}, min=0, max=100, step=1, description='{name}')"
    if kind == "float":
        return f"FloatSlider(value={value}, min=0.0, max=1.0, step=0.1, description='{name}')"
    return f"Text(value={value}, description='{name}')"


def _interactive_markdown(model: Dict[str, Any]) -> Dict[str, Any]:
    """Explanation cell shown before the interactive widgets."""
    tunable, _ = _split_parameters(model)
    if not tunable:
        return _markdown(
            "## 🎛️ Ejemplo interactivo\n\n"
            "Este modelo no declara parámetros escalares, por lo que el ejemplo "
            "interactivo se limita a construir el objeto.\n"
        )
    controls = ", ".join(f"`{parameter['name']}`" for parameter in tunable)
    return _markdown(
        "## 🎛️ Ejemplo interactivo\n\n"
        f"Mueve los controles {controls} para reconfigurar el modelo.\n\n"
        "> Requiere `ipywidgets` (`pip install ipywidgets`).\n"
    )


def _interactive_source(model: Dict[str, Any]) -> str:
    """Build a widget driven cell from the real signature."""
    tunable, fixed = _split_parameters(model)

    required = [p for p in fixed if p["required"]]
    optional = [p for p in fixed if not p["required"] and _is_safe_literal(p["default"])]

    setup: List[str] = []
    for parameter in required:
        name = parameter["name"]
        setup.append(f"{name} = {parameter['sample']}  # {parameter['type']}")

    body = "\n".join(setup)
    imports = [model["import_statement"]]
    if "np." in body:
        imports.append("import numpy as np")
    if "pd." in body:
        imports.append("import pandas as pd")

    widget_names = [p["name"] for p in tunable]
    widget_imports: List[str] = []
    if tunable:
        kinds = sorted({p["widget"] for p in tunable})
        widget_imports.append(
            "from ipywidgets import " + ", ".join([*_widget_classes(kinds), "interact"])
        )

    lines = [f"# Ejemplo interactivo de {model['title']}"]
    lines.append("# Generado automáticamente a partir de la firma real de la clase.")
    lines.extend(imports)
    lines.extend(widget_imports)
    lines.append("")
    if setup:
        lines.append("# 1) Datos de entrada requeridos: ajústalos a tu problema")
        lines.extend(setup)
        lines.append("")

    lines.append("# 2) Controles interactivos")
    lines.append("widgets = {")
    for parameter in tunable:
        lines.append(f"    {parameter['name']!r}: {_widget_expression(parameter)},")
    lines.append("}")
    lines.append("")

    lines.append("def run_example(**overrides):")
    lines.append(f'    """Construye {model["id"]} con los valores elegidos."""')
    lines.append("    values = {")
    for parameter in tunable:
        lines.append(f"        {parameter['name']!r}: overrides[{parameter['name']!r}],")
    for parameter in required:
        lines.append(f"        {parameter['name']!r}: {parameter['name']},")
    for parameter in optional:
        lines.append(f"        {parameter['name']!r}: {parameter['default']},")
    lines.append("    }")
    lines.append(f"    model = {model['id']}(**values)")
    calls = model.get("safe_calls") or []
    if calls:
        lines.append(f"    # Sugerencias: {', '.join(calls)}")
    lines.append("    print('✅ Modelo construido:', type(model).__name__)")
    lines.append("    return model")
    lines.append("")

    if widget_names:
        lines.append("interact(run_example, **widgets);")
    else:
        lines.append("run_example();")
    return "\n".join(lines)


def _widget_classes(kinds: Sequence[str]) -> List[str]:
    """Ipywidgets class names for the requested control kinds."""
    mapping = {
        "int": "IntSlider",
        "float": "FloatSlider",
        "bool": "Checkbox",
        "str": "Text",
    }
    return [mapping[kind] for kind in kinds if kind in mapping]


def apply_parameter_values(model: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of ``model`` whose parameter defaults are the given values.

    Overriding ``default``/``sample`` with ``repr(value)`` keeps every generator
    below untouched: the widget expressions, the example and the parameter table
    all read those two keys.  Variadic parameters and unknown names are ignored.
    """
    overrides = {key: value for key, value in parameters.items() if value is not None}
    if not overrides:
        return model
    patched = dict(model)
    patched["parameters"] = [dict(parameter) for parameter in model["parameters"]]
    for parameter in patched["parameters"]:
        name = parameter["name"]
        if parameter.get("variadic") or name not in overrides:
            continue
        literal = repr(overrides[name])
        parameter["default"] = literal
        parameter["sample"] = literal
    return patched


def build_notebook(
    model: Dict[str, Any],
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the complete notebook document for ``model``.

    Parameters
    ----------
    model :
        A catalog entry produced by :func:`model_catalog.build_catalog`.
    parameters :
        Values already validated by the playground (``coerce_playground_value``).
        They become the defaults of the generated ``ipywidgets`` controls, which
        is what lets ``POST /api/notebook`` bake a playground run into the
        notebook.  ``None`` keeps the documented defaults.

    Returns
    -------
    dict
        A notebook following nbformat 4.5.
    """
    if parameters:
        model = apply_parameter_values(model, parameters)
    notebook = _empty_notebook(model)
    if parameters:
        notebook["metadata"]["likelihood"]["parameters"] = dict(parameters)
    cells: List[Dict[str, Any]] = notebook["cells"]
    cells.append(_title_cell(model))
    cells.append(_description_cell(model))
    if model["parameters"]:
        cells.append(_parameters_cell(model))
    if model["methods"]:
        cells.append(_methods_cell(model))
    cells.append(_signature_cell(model))
    cells.append(_usage_cell(model))
    cells.append(_code(model["example"], tags=["example"]))
    cells.append(_interactive_markdown(model))
    cells.append(_code(_interactive_source(model), tags=["interactive"]))
    return notebook


def validate_notebook(notebook: Dict[str, Any]) -> Dict[str, Any]:
    """Validate ``notebook`` with ``nbformat`` when the package is available."""
    try:
        import nbformat
    except ImportError:  # pragma: no cover - nbformat is optional
        return notebook
    node = nbformat.from_dict(notebook)
    nbformat.validate(node)
    return node


def write_notebook(model: Dict[str, Any], output_dir: Path) -> Path:
    """Write the notebook of ``model`` into ``output_dir`` and return its path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook = validate_notebook(build_notebook(model))
    path = output_dir / f"{model['id']}.ipynb"
    path.write_text(json.dumps(notebook, indent=_INDENT, ensure_ascii=False) + "\n", "utf-8")
    return path


def generate_notebook(model_id: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Optional[Path]:
    """Generate the notebook for a single model id."""
    catalog = build_catalog()
    model = find_model(catalog, model_id)
    if model is None:
        return None
    return write_notebook(model, output_dir)


def generate_category(category: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> List[Path]:
    """Generate the notebooks of every model in ``category``."""
    grouped = models_by_category(build_catalog())
    if category not in grouped:
        raise KeyError(category)
    return [write_notebook(model, output_dir) for model in grouped[category]]


def generate_all_notebooks(output_dir: Path = DEFAULT_OUTPUT_DIR) -> List[Path]:
    """Generate the notebook of every model in the catalog."""
    return [write_notebook(model, output_dir) for model in build_catalog()]


def _print_catalog(catalog: Sequence[Dict[str, Any]]) -> None:
    """Print the catalog grouped by category."""
    grouped = models_by_category(catalog)
    for category in sorted(grouped):
        label, icon = category_meta(category)
        print(f"\n{icon} {label} ({category})")
        for model in grouped[category]:
            print(f"    {model['id']:<20} {model['module']}")
    counts = summarize(catalog)
    print(f"\n{counts['total']} modelos en {len(grouped)} categorías")


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Genera notebooks de Jupyter a partir del catálogo de modelos."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--single", metavar="MODELO", help="genera solo este modelo")
    selection.add_argument("--category", metavar="CATEGORIA", help="genera una categoría")
    selection.add_argument("--all", action="store_true", help="genera todos los modelos")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directorio de salida (por defecto: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=None,
        help="directorio del paquete de modelos a inspeccionar",
    )
    parser.add_argument("--list", action="store_true", help="muestra el catálogo y sale")
    arguments = parser.parse_args(argv)

    catalog = build_catalog(arguments.package_dir)
    if not catalog:
        print("No se encontraron modelos en el paquete.", file=sys.stderr)
        return 1

    if arguments.list:
        _print_catalog(catalog)
        return 0

    output_dir = Path(arguments.output)
    if arguments.single:
        model = find_model(catalog, arguments.single)
        if model is None:
            available = ", ".join(model["id"] for model in catalog)
            print(
                f"Modelo desconocido: {arguments.single}\nDisponibles: {available}", file=sys.stderr
            )
            return 2
        targets = [model]
    elif arguments.category:
        grouped = models_by_category(catalog)
        if arguments.category not in grouped:
            available = ", ".join(sorted(grouped))
            print(
                f"Categoría desconocida: {arguments.category}\nDisponibles: {available}",
                file=sys.stderr,
            )
            return 2
        targets = grouped[arguments.category]
    else:
        targets = catalog

    for model in targets:
        path = write_notebook(model, output_dir)
        print(f"✓ {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")

    print(f"\n✅ {len(targets)} notebooks generados en {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
