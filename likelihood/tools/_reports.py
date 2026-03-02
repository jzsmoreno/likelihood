import datetime
from html import escape
from typing import Any

from IPython.display import HTML, display


def generate_html_pipeline(
    data_dict: Any,
    save_to_file: bool = False,
    file_name: str = "data_processing_report.html",
):
    """
    Generates an HTML report for a data processing pipeline with an improved
    layout using a tabbed interface, better typography, animations, and
    accessible markup.
    """

    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --c-bg: #f1f5f9;
            --c-surface: #ffffff;
            --c-primary: #0d9488;
            --c-primary-hover: #0f766e;
            --c-accent: #6366f1;
            --c-success: #22c55e;
            --c-text: #0f172a;
            --c-text-muted: #64748b;
            --c-border: #e2e8f0;
            --c-row-alt: #f8fafc;
            --c-highlight: #f0fdfa;
            --radius: 10px;
            --shadow: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
            --shadow-lg: 0 4px 16px rgba(0,0,0,.08);
            --transition: .2s ease;
        }

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: var(--c-bg);
            color: var(--c-text);
            padding: 2rem;
            font-size: 14px;
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }

        /* ── Header ── */
        .report-header {
            background: linear-gradient(135deg, #0d9488 0%, #6366f1 100%);
            color: #fff;
            padding: 2rem 2.5rem;
            border-radius: var(--radius);
            margin-bottom: 2rem;
            box-shadow: var(--shadow-lg);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }
        .report-header h1 { font-size: 1.75rem; font-weight: 700; letter-spacing: -.02em; }
        .report-header .subtitle { opacity: .85; font-size: .85rem; font-weight: 400; }

        /* ── Stats Bar ── */
        .stats-bar {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .stat-card {
            background: var(--c-surface);
            border-radius: var(--radius);
            padding: 1.25rem 1.5rem;
            box-shadow: var(--shadow);
            border-left: 4px solid var(--c-primary);
            transition: transform var(--transition), box-shadow var(--transition);
        }
        .stat-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-lg); }
        .stat-card .label { font-size: .75rem; text-transform: uppercase; letter-spacing: .05em; color: var(--c-text-muted); margin-bottom: .25rem; }
        .stat-card .value { font-size: 1.5rem; font-weight: 700; color: var(--c-primary); }

        /* ── Tabs ── */
        .tabs-wrapper {
            background: var(--c-surface);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            overflow: hidden;
        }
        .tab-bar {
            display: flex;
            border-bottom: 2px solid var(--c-border);
            background: var(--c-row-alt);
        }
        .tab-bar button {
            flex: 1;
            padding: 1rem 1.25rem;
            border: none;
            background: transparent;
            font: inherit;
            font-weight: 600;
            font-size: .9rem;
            color: var(--c-text-muted);
            cursor: pointer;
            position: relative;
            transition: color var(--transition), background var(--transition);
        }
        .tab-bar button:hover { background: var(--c-highlight); color: var(--c-primary); }
        .tab-bar button.active { color: var(--c-primary); background: var(--c-surface); }
        .tab-bar button.active::after {
            content: '';
            position: absolute;
            bottom: -2px; left: 0; right: 0;
            height: 3px;
            background: var(--c-primary);
            border-radius: 3px 3px 0 0;
        }
        .tab-panel { display: none; padding: 2rem; animation: fadeIn .25s ease; }
        .tab-panel.active { display: block; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

        /* ── Section Titles ── */
        .section-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--c-text);
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            gap: .5rem;
        }
        .section-title .icon { font-size: 1.3rem; }

        /* ── Tables ── */
        table { width: 100%; border-collapse: collapse; font-size: .875rem; }
        th, td { padding: .75rem 1rem; text-align: left; border-bottom: 1px solid var(--c-border); vertical-align: top; }
        th {
            background: var(--c-row-alt);
            font-weight: 600;
            font-size: .75rem;
            text-transform: uppercase;
            letter-spacing: .04em;
            color: var(--c-text-muted);
            position: sticky; top: 0;
        }
        tbody tr:hover { background: var(--c-highlight); }
        .nested-table { font-size: .8rem; margin-top: .5rem; border: 1px solid var(--c-border); border-radius: 6px; overflow: hidden; }
        .nested-table th { background: #f0fdf4; }

        /* ── Accordion Steps ── */
        .step-accordion { margin-bottom: .75rem; border: 1px solid var(--c-border); border-radius: var(--radius); overflow: hidden; transition: box-shadow var(--transition); }
        .step-accordion:hover { box-shadow: var(--shadow); }
        .step-accordion summary {
            display: flex;
            align-items: center;
            gap: .75rem;
            padding: 1rem 1.25rem;
            font-weight: 600;
            font-size: .95rem;
            color: var(--c-text);
            cursor: pointer;
            list-style: none;
            background: var(--c-row-alt);
            transition: background var(--transition);
        }
        .step-accordion summary::-webkit-details-marker { display: none; }
        .step-accordion summary:hover { background: var(--c-highlight); }
        .step-accordion[open] summary { background: var(--c-highlight); border-bottom: 1px solid var(--c-border); }
        .step-number {
            display: inline-flex; align-items: center; justify-content: center;
            width: 28px; height: 28px; border-radius: 50%;
            background: var(--c-primary); color: #fff;
            font-size: .8rem; font-weight: 700; flex-shrink: 0;
        }
        .chevron {
            margin-left: auto;
            transition: transform .2s;
            color: var(--c-text-muted);
            font-size: .8rem;
        }
        .step-accordion[open] .chevron { transform: rotate(90deg); }
        .step-body { padding: 1.25rem 1.5rem; }
        .step-desc { color: var(--c-text-muted); margin-bottom: 1rem; font-size: .875rem; line-height: 1.5; }
        .sub-section-title { font-weight: 600; font-size: .85rem; color: var(--c-primary-hover); margin: 1rem 0 .5rem; display: flex; align-items: center; gap: .35rem; }

        /* ── Badge ── */
        .badge {
            display: inline-block;
            padding: .15rem .55rem;
            border-radius: 999px;
            font-size: .7rem;
            font-weight: 600;
            background: #dbeafe;
            color: #1e40af;
        }

        /* ── Footer ── */
        .report-footer { text-align: center; color: var(--c-text-muted); font-size: .75rem; margin-top: 2.5rem; padding-top: 1.5rem; border-top: 1px solid var(--c-border); }

        /* ── Responsive ── */
        @media (max-width: 640px) {
            body { padding: 1rem; font-size: 13px; }
            .report-header { padding: 1.5rem; }
            .report-header h1 { font-size: 1.35rem; }
            .tab-bar button { padding: .75rem .5rem; font-size: .8rem; }
            .tab-panel { padding: 1.25rem; }
            th, td { padding: .5rem .6rem; }
        }
    </style>
    """

    js = """
    <script>
        function openTab(evt, id) {
            document.querySelectorAll('.tab-panel').forEach(p => { p.style.display = 'none'; p.classList.remove('active'); });
            document.querySelectorAll('.tab-bar button').forEach(b => b.classList.remove('active'));
            const panel = document.getElementById(id);
            panel.style.display = 'block';
            // Force reflow for animation restart
            void panel.offsetWidth;
            panel.classList.add('active');
            evt.currentTarget.classList.add('active');
        }
        window.addEventListener('DOMContentLoaded', () => document.querySelector('.tab-bar button').click());
    </script>
    """

    # ── Helpers ──

    def render_value(val):
        if isinstance(val, dict):
            return dict_to_table(val, nested=True)
        if isinstance(val, list):
            if all(isinstance(item, (str, int, float)) for item in val):
                return ", ".join(f"<span class='badge'>{escape(str(x))}</span>" for x in val)
            return (
                "<ul style='margin:.25rem 0;padding-left:1.2rem'>"
                + "".join(f"<li>{render_value(v)}</li>" for v in val)
                + "</ul>"
            )
        return escape(str(val))

    def dict_to_table(d, nested=False):
        if not isinstance(d, dict):
            d = {"Error": "Data not available or incorrect format"}
        cls = "nested-table" if nested else ""
        rows = ""
        for key, val in d.items():
            if val is None or (isinstance(val, (str, list, dict)) and not val):
                continue
            rows += (
                f"<tr><td><strong>{escape(str(key))}</strong></td><td>{render_value(val)}</td></tr>"
            )
        return f"<table class='{cls}'><thead><tr><th>Key</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>"

    # ── Quick Stats ──

    initial = data_dict.get("initial_dataset", {})
    final = data_dict.get("final_dataset", {})
    steps = data_dict.get("processing_steps", [])

    def extract_shape(ds):
        shape = ds.get("shape") or ds.get("Shape")
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            return shape
        return None, None

    init_rows, init_cols = extract_shape(initial)
    final_rows, final_cols = extract_shape(final)

    stats_html = '<div class="stats-bar">'
    stat_items = [
        ("Pipeline Steps", str(len(steps))),
        ("Initial Rows", str(init_rows) if init_rows is not None else "—"),
        ("Final Rows", str(final_rows) if final_rows is not None else "—"),
        ("Final Columns", str(final_cols) if final_cols is not None else "—"),
    ]
    for label, value in stat_items:
        stats_html += f'<div class="stat-card"><div class="label">{label}</div><div class="value">{value}</div></div>'
    stats_html += "</div>"

    # ── Steps ──

    steps_html = ""
    for i, step in enumerate(steps):
        name = escape(step.get("step_name", "Unnamed Step"))
        desc = escape(step.get("description", "No description provided"))

        params_block = ""
        params_data = step.get("parameters", {})
        if params_data:
            params_block = f'<div class="sub-section-title">⚙️ Parameters</div>{dict_to_table(params_data, nested=True)}'

        output_info = {
            "Output Shape": step.get("output_shape", "N/A"),
            "Input Columns": step.get("input_columns", "N/A"),
            "Output Columns": step.get("output_columns", "N/A"),
            "Output Dtypes": step.get("output_dtypes", "N/A"),
            "Category Columns": step.get("unique_categories", "N/A"),
        }
        # Filter out N/A-only entries for cleaner output
        output_info = {k: v for k, v in output_info.items() if v != "N/A"}

        output_block = ""
        if output_info:
            output_block = f'<div class="sub-section-title">📊 Output Info</div>{dict_to_table(output_info, nested=True)}'

        steps_html += f"""
        <details class="step-accordion">
            <summary>
                <span class="step-number">{i + 1}</span>
                {name}
                <span class="chevron">▶</span>
            </summary>
            <div class="step-body">
                <p class="step-desc">{desc}</p>
                {params_block}
                {output_block}
            </div>
        </details>
        """

    # ── Assemble ──

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Processing Report</title>
    {css}
    {js}
</head>
<body>

<div class="report-header">
    <div>
        <h1>📈 Data Processing Report</h1>
        <div class="subtitle">Generated {current_time}</div>
    </div>
</div>

{stats_html}

<div class="tabs-wrapper">
    <div class="tab-bar" role="tablist">
        <button role="tab" onclick="openTab(event,'tab-initial')">📁 Initial Dataset</button>
        <button role="tab" onclick="openTab(event,'tab-steps')">🔧 Processing Steps</button>
        <button role="tab" onclick="openTab(event,'tab-final')">✅ Final Dataset</button>
    </div>

    <div id="tab-initial" class="tab-panel" role="tabpanel">
        <div class="section-title"><span class="icon">📁</span> Initial Dataset Overview</div>
        {dict_to_table(initial)}
    </div>

    <div id="tab-steps" class="tab-panel" role="tabpanel">
        <div class="section-title"><span class="icon">🔧</span> Pipeline Steps ({len(steps)})</div>
        {steps_html if steps_html else '<p style="color:var(--c-text-muted)">No processing steps recorded.</p>'}
    </div>

    <div id="tab-final" class="tab-panel" role="tabpanel">
        <div class="section-title"><span class="icon">✅</span> Final Dataset Overview</div>
        {dict_to_table(final)}
    </div>
</div>

<div class="report-footer">Pipeline Report &middot; {current_time}</div>

</body>
</html>"""

    if save_to_file:
        try:
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"✅ Report saved to '{file_name}'")
        except NameError:
            return f"Report content saved to '{file_name}'"
    else:
        try:
            display(HTML(html))
        except (ImportError, NameError):
            return html
