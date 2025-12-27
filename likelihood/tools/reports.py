import datetime
from html import escape
from typing import Any

from IPython.display import HTML, display


def generate_html_pipeline(
    data_dict: Any,
    save_to_file: bool = False,
    file_name: str = "data_processing_report_improved.html",
):
    """
    Generates an HTML report for a data processing pipeline with an improved
    layout using a tabbed interface to reduce vertical scroll.
    """
    css_js = """
    <style>
        /* Existing Styles (omitted for brevity, keep the original content) */
        :root {
            --primary: #0d9488;
            --primary-dark: #0f766e;
            --success: #10b981;
            --accent: #3b82f6;
            --card-bg: #ffffff;
            --shadow-sm: 0 2px 6px rgba(0, 0, 0, 0.03);
            --border-radius-md: 8px;
            --font-family: 'Roboto', sans-serif;
        }

        * {
            box-sizing: border-box;
        }

        body {
            font-family: var(--font-family);
            background: #f8fafc;
            color: #1e293b;
            margin: 0;
            padding: 1.5rem;
            font-size: 14px;
        }

        h2 {
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: white;
            text-align: center;
            padding: 1.2rem;
            border-radius: var(--border-radius-md);
            font-weight: 700;
            font-size: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }

        section {
            background: var(--card-bg);
            border-radius: var(--border-radius-md);
            padding: 1.5rem;
            box-shadow: var(--shadow-sm);
            margin-bottom: 2rem;
        }

        h3 {
            color: var(--primary-dark);
            font-weight: 600;
            font-size: 1.4rem;
            border-left: 5px solid var(--success);
            padding-left: 1rem;
            margin-bottom: 1rem;
            transition: color 0.3s ease;
        }

        h3:hover {
            color: var(--primary);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            margin: 1rem 0;
        }

        th, td {
            padding: 0.8rem 1rem;
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
            margin-top: 0.8rem;
        }

        details {
            margin-bottom: 1rem;
            padding: 0.8rem;
            background: #f9f9f9;
            border-radius: var(--border-radius-md);
            transition: background 0.3s ease;
        }

        details[open] {
            background: #f0fdf4;
        }

        summary {
            font-weight: 700;
            font-size: 1.1rem;
            color: var(--primary-dark);
            cursor: pointer;
            list-style: none; /* Hide default marker */
            transition: color 0.3s ease;
        }

        summary::-webkit-details-marker {
            display: none; /* For Chrome/Safari */
        }

        summary::before {
            content: "▶";
            margin-right: 6px;
            color: var(--success);
            font-size: 1rem;
            display: inline-block;
            transition: transform 0.2s;
        }
        
        details[open] summary::before {
            content: "▼";
        }

        summary:hover {
            color: var(--primary);
        }
        
        .tabbed-interface {
            display: flex;
            flex-direction: column;
            background: var(--card-bg);
            border-radius: var(--border-radius-md);
            box-shadow: var(--shadow-sm);
            padding: 0;
            margin-bottom: 2rem;
        }

        .tab-buttons {
            display: flex;
            border-bottom: 2px solid #e2e8f0;
            flex-wrap: wrap; /* Allows wrapping on smaller screens */
        }

        .tab-button {
            padding: 1rem 1.5rem;
            cursor: pointer;
            border: none;
            background: transparent;
            font-weight: 600;
            color: #475569;
            font-size: 1rem;
            transition: color 0.3s, border-bottom 0.3s;
            flex-grow: 1; /* Makes buttons take up equal space */
            text-align: center;
        }

        .tab-button.active, .tab-button:hover {
            color: var(--primary);
            border-bottom: 3px solid var(--primary);
        }

        .tab-content {
            padding: 1.5rem;
        }

        .tab-pane {
            display: none;
        }

        .tab-pane.active {
            display: block;
        }

        /* --- END NEW TABBED LAYOUT STYLES --- */

        @media (max-width: 768px) {
            body {
                font-size: 13px;
            }

            h2 {
                font-size: 1.6rem;
                padding: 1rem;
            }

            h3 {
                font-size: 1.3rem;
            }

            table, .nested-table {
                font-size: 12px;
            }

            section {
                padding: 1rem;
            }
            
            .tab-button {
                padding: 0.8rem 1rem;
                font-size: 0.9rem;
            }
        }
    </style>
    <script>
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tab-pane");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
                tabcontent[i].classList.remove("active");
            }
            tablinks = document.getElementsByClassName("tab-button");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].classList.remove("active");
            }
            document.getElementById(tabName).style.display = "block";
            document.getElementById(tabName).classList.add("active");
            evt.currentTarget.classList.add("active");
            // Scroll to the top of the tabbed interface container
            document.querySelector('.tabbed-interface').scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        // Set 'Initial Dataset' as the default active tab on load
        window.onload = function() {
            document.querySelector('.tab-button').click();
        };
    </script>
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
        table_class = "nested-table" if nested else "main-table"
        html += f"<table class='{table_class}'>"
        html += "<thead><tr><th>Key</th><th>Value</th></tr></thead><tbody>"
        if not isinstance(d, dict):
            d = {"Error": "Data not available or incorrect format"}

        for key, val in d.items():
            if val is None or (isinstance(val, (str, list, dict)) and not val):
                continue
            key_html = escape(str(key))
            val_html = render_value(val)
            html += f"<tr><td>{key_html}</td><td>{val_html}</td></tr>"
        html += "</tbody></table>"
        return html

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data Processing Report</title>
        {css_js}
    </head>
    <body>
        <h2>📈 Data Processing Report</h2>
    """
    html_content += """
        <div class="tabbed-interface">
            <div class="tab-buttons">
                <button class="tab-button" onclick="openTab(event, 'initial_dataset_tab')">📁 Initial Dataset</button>
                <button class="tab-button" onclick="openTab(event, 'processing_steps_tab')">🔧 Processing Steps</button>
                <button class="tab-button" onclick="openTab(event, 'final_dataset_tab')">✅ Final Dataset</button>
            </div>
            <div class="tab-content">
    """
    html_content += """
                <div id="initial_dataset_tab" class="tab-pane">
                    <h3>📁 Initial Dataset Overview</h3>
    """
    html_content += dict_to_table(data_dict.get("initial_dataset", {}))
    html_content += """
                </div>
    """
    html_content += """
                <div id="processing_steps_tab" class="tab-pane">
                    <h3>🔧 Pipeline Steps Breakdown</h3>
    """
    for i, step in enumerate(data_dict.get("processing_steps", [])):
        html_content += f"<details>"
        html_content += (
            f"<summary>Step {i + 1}: {escape(step.get('step_name', 'Unnamed Step'))}</summary>"
        )
        html_content += f"<p><strong>Description:</strong> {escape(step.get('description', 'No description provided'))}</p>"

        html_content += (
            f"<div style='margin-left: 1rem; border-left: 3px solid #e0f2fe; padding-left: 1rem;'>"
        )

        params_data = step.get("parameters", {})
        if params_data:
            html_content += "<h4>⚙️ Parameters</h4>"
            html_content += dict_to_table(params_data, nested=True)
        else:
            html_content += "<p><em>No parameters recorded.</em></p>"

        output_info = {
            "Output Shape": step.get("output_shape", "N/A"),
            "Input Columns": step.get("input_columns", "N/A"),
            "Output Columns": step.get("output_columns", "N/A"),
            "Output Dtypes": step.get("output_dtypes", "N/A"),
            "Category Columns": step.get("unique_categories", "N/A"),
        }
        html_content += "<h4>📊 Output Information</h4>"
        html_content += dict_to_table(output_info, nested=True)

        html_content += "</div>"
        html_content += "</details>"
    html_content += """
                </div>
    """

    html_content += """
                <div id="final_dataset_tab" class="tab-pane">
                    <h3>✅ Final Dataset Overview</h3>
    """
    html_content += dict_to_table(data_dict.get("final_dataset", {}))
    html_content += """
                </div>
            </div> </div> """
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_content += f"""
        <p style="text-align: center; color: #94a3b8; font-size: 0.8rem; margin-top: 3rem;">
            Report Generated by Pipeline at {current_time_str}
        </p>
    </body>
    </html>
    """
    if save_to_file:
        try:
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"✅ Report saved to '{file_name}'")
        except NameError:
            return f"Report content saved to '{file_name}'"
    else:
        try:
            display(HTML(html_content))
        except (ImportError, NameError):
            return html_content
