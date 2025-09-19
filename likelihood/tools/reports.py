from html import escape
from IPython.display import HTML, display


def generate_html_pipeline(data_dict, save_to_file=False, file_name="data_processing_report.html"):
    css_js = """
    <style>
        :root {
            --primary: #0d9488;
            --primary-dark: #0f766e;
            --success: #10b981;
            --accent: #3b82f6;
            --card-bg: #ffffff;
            --shadow-sm: 0 2px 6px rgba(0, 0, 0, 0.03);
            --border-radius-md: 6px;
        }

        * {
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #f8fafc;
            color: #1e293b;
            margin: 0;
            padding: 1rem;
            font-size: 14px;
        }

        h2 {
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: white;
            text-align: center;
            padding: 1rem;
            border-radius: var(--border-radius-md);
            font-weight: 600;
            font-size: 1.5rem;
            margin-bottom: 1.5rem;
        }

        section {
            background: var(--card-bg);
            border-radius: var(--border-radius-md);
            padding: 1rem;
            box-shadow: var(--shadow-sm);
            margin-bottom: 1.2rem;
        }

        h3 {
            color: var(--primary-dark);
            font-weight: 600;
            font-size: 1.2rem;
            border-left: 4px solid var(--success);
            padding-left: 0.8rem;
            margin: 1rem 0 0.8rem;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin: 0.5rem 0 1rem;
        }

        th, td {
            padding: 0.5rem 0.75rem;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
            vertical-align: top;
        }

        thead {
            background-color: #f0fdf4;
        }

        tbody tr:nth-child(odd) {
            background-color: #f9fafb;
        }

        tbody tr:hover {
            background-color: #e0f2fe;
        }

        .nested-table {
            font-size: 12px;
            margin-top: 0.5rem;
        }

        details {
            margin-bottom: 0.8rem;
            padding: 0.5rem 0.8rem;
            background: #f9f9f9;
            border-radius: var(--border-radius-md);
        }

        summary {
            font-weight: 600;
            font-size: 1rem;
            color: var(--primary-dark);
            cursor: pointer;
        }

        summary::before {
            content: "▶";
            margin-right: 6px;
            color: var(--success);
            font-size: 0.9rem;
        }

        @media (max-width: 768px) {
            body {
                font-size: 13px;
            }

            h2 {
                font-size: 1.3rem;
                padding: 0.8rem;
            }

            h3 {
                font-size: 1.1rem;
            }

            table, .nested-table {
                font-size: 12px;
            }
        }
    </style>
    """

    def render_value(val):
        if isinstance(val, dict):
            return dict_to_table(val, nested=True)
        elif isinstance(val, list):
            if all(isinstance(item, (str, int, float)) for item in val):
                return ", ".join(escape(str(x)) for x in val)
            else:
                return "<ul>" + "".join(f"<li>{render_value(v)}</li>" for v in val) + "</ul>"
        else:
            return escape(str(val))

    def dict_to_table(d, title=None, nested=False):
        html = ""
        if title and not nested:
            html += f"<h4>{escape(title)}</h4>"
        table_class = "nested-table" if nested else "table"
        html += f"<table class='{table_class}'>"
        html += "<thead><tr><th>Key</th><th>Value</th></tr></thead><tbody>"
        for key, val in d.items():
            key_html = escape(str(key))
            val_html = render_value(val)
            html += f"<tr><td>{key_html}</td><td>{val_html}</td></tr>"
        html += "</tbody></table>"
        return html

    html_content = css_js
    html_content += "<h2>📈 Data Processing Report</h2>"

    html_content += "<section>"
    html_content += "<h3>📁 Initial Dataset</h3>"
    html_content += dict_to_table(data_dict["initial_dataset"])
    html_content += "</section>"

    html_content += "<section>"
    html_content += "<h3>🔧 Processing Steps</h3>"
    for i, step in enumerate(data_dict["processing_steps"]):
        html_content += "<details open>"
        html_content += f"<summary>Step {i + 1}: {escape(step['step_name'])}</summary>"
        html_content += f"<p><strong>Description:</strong> {escape(step['description'])}</p>"
        html_content += dict_to_table(step["parameters"], title="Parameters", nested=True)
        html_content += dict_to_table(
            {
                "Output Shape": step["output_shape"],
                "Input Columns": step["input_columns"],
                "Output Columns": step["output_columns"],
                "Output Dtypes": step["output_dtypes"],
            },
            title="Output Info",
            nested=True,
        )
        html_content += "</details>"
    html_content += "</section>"

    html_content += "<section>"
    html_content += "<h3>✅ Final Dataset</h3>"
    html_content += dict_to_table(data_dict["final_dataset"])
    html_content += "</section>"

    if save_to_file:
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"✅ Report saved to '{file_name}'")
    else:
        display(HTML(html_content))
