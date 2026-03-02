import datetime
import json
import math
import re
from html import escape
from typing import Any, Dict, List

from IPython.display import HTML, display


def generate_html_pipeline(
    data_dict: Any,
    save_to_file: bool = False,
    file_name: str = "data_processing_report.html",
):
    """
    Generates an HTML report for a data processing pipeline with:
      - Tabbed interface (Initial / Steps / Final / Visualizations)
      - Stat cards summary bar
      - Automatic chart generation from sample_data:
          * Histograms / Box plots / Violin plots for numeric columns
          * Bar / Donut / Pareto charts for categorical columns
          * Ordered Range Bar / Cumulative Step / Proportion charts for one-hot range columns
          * Correlation heatmap (Pearson / Spearman / Kendall)
      - Interactive chart-type toggle per column

    Expected keys in data_dict:
        initial_dataset : dict
        processing_steps: list
        final_dataset   : dict
        sample_data     : dict - {"columns": [...], "rows": [[...], ...]}
                                  or list-of-dicts [{col: val, ...}, ...]
    """
    # ── Normalise sample_data ──
    raw_sample = data_dict.get("sample_data")
    columns: List[str] = []
    col_values: Dict[str, list] = {}

    if isinstance(raw_sample, dict) and "columns" in raw_sample and "rows" in raw_sample:
        columns = raw_sample["columns"]
        for ci, col in enumerate(columns):
            col_values[col] = [row[ci] for row in raw_sample["rows"] if ci < len(row)]
    elif isinstance(raw_sample, list) and raw_sample and isinstance(raw_sample[0], dict):
        columns = list(raw_sample[0].keys())
        for col in columns:
            col_values[col] = [row.get(col) for row in raw_sample]

    # ── Classify columns ──
    # Detect one-hot range columns:
    #   1. Column name contains a numeric range with "-" (e.g. "18-24",
    #      "age_0-50000", "income 100.5-200.5", "prefix_18-24")
    #   2. Column values are exclusively 0/1 (binary one-hot)
    # Grouping is done primarily by delta (bin width).
    range_extract = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    # All detected one-hot range columns before grouping
    _onehot_detected: List[tuple] = []  # (col, prefix, lo, hi, delta)

    for col in columns:
        m = range_extract.search(col)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            if lo < hi:
                # Check if values are binary (only 0 and 1)
                vals = [v for v in col_values[col] if v is not None]
                unique_vals = set()
                for v in vals:
                    if isinstance(v, (int, float)):
                        unique_vals.add(float(v))
                    elif isinstance(v, bool):
                        unique_vals.add(1.0 if v else 0.0)

                is_binary = len(unique_vals) > 0 and unique_vals <= {0.0, 1.0}

                if is_binary:
                    # Extract prefix: everything before the numeric range
                    prefix_part = col[: m.start()].rstrip("_ ")
                    prefix = prefix_part if prefix_part else ""
                    delta = round(hi - lo, 6)
                    _onehot_detected.append((col, prefix, lo, hi, delta))
                    continue

        # Not a one-hot range column — classify normally
        nums = [
            v
            for v in col_values[col]
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))
        ]
        if len(nums) >= 3:
            numeric_cols.append(col)
        elif col_values[col]:
            categorical_cols.append(col)

    # Group one-hot columns primarily by delta, then by prefix.
    # Columns with the same delta AND same prefix belong together.
    # Columns with the same delta but no prefix (bare ranges) are
    # grouped together if they form a coherent set.
    from collections import defaultdict

    onehot_groups: Dict[str, List[tuple]] = {}
    _by_delta_prefix: Dict[tuple, List[tuple]] = defaultdict(list)

    for col, prefix, lo, hi, delta in _onehot_detected:
        key = (delta, prefix)
        _by_delta_prefix[key].append((col, lo, hi))

    for (delta, prefix), members in _by_delta_prefix.items():
        if len(members) >= 2:
            members.sort(key=lambda x: x[1])
            if prefix:
                group_label = prefix
            else:
                # Bare ranges — label by delta
                delta_fmt = str(int(delta)) if delta == int(delta) else f"{delta}"
                group_label = f"range (Δ={delta_fmt})"
            # Deduplicate labels
            base_label = group_label
            counter = 2
            while group_label in onehot_groups:
                group_label = f"{base_label} #{counter}"
                counter += 1
            onehot_groups[group_label] = members
        else:
            # Single column with this delta/prefix — not a group,
            # fall back to numeric (it's binary but still numeric)
            for col, lo, hi in members:
                numeric_cols.append(col)

    # Collect all one-hot column names for correlation analysis
    # (binary 0/1 columns should participate in correlations)
    onehot_col_names: List[str] = []
    for bins in onehot_groups.values():
        for col, lo, hi in bins:
            onehot_col_names.append(col)

    # Correlation columns = numeric + one-hot (binary)
    corr_cols = numeric_cols + onehot_col_names

    col_values_json = json.dumps(
        {col: [v for v in col_values[col] if v is not None] for col in columns}
    )
    numeric_cols_json = json.dumps(numeric_cols)
    categorical_cols_json = json.dumps(categorical_cols)
    corr_cols_json = json.dumps(corr_cols)
    # Build a safe-key mapping for JS: group_label → safe_id and data
    # We use safe_id as the JS key to avoid issues with special characters
    onehot_js_map = {}  # safe_id → {label, bins}
    onehot_safe_ids = {}  # group_label → safe_id
    for group_label, bins in onehot_groups.items():
        safe_id = "".join(c if c.isalnum() else "_" for c in group_label)
        # Ensure unique
        base = safe_id
        counter = 2
        while safe_id in onehot_js_map:
            safe_id = f"{base}_{counter}"
            counter += 1
        onehot_safe_ids[group_label] = safe_id
        onehot_js_map[safe_id] = {
            "label": group_label,
            "bins": [{"col": c, "lo": lo, "hi": hi} for c, lo, hi in bins],
        }
    onehot_groups_json = json.dumps(onehot_js_map)

    # ── CSS ──
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        :root {
            --c-bg:#f1f5f9;--c-surface:#fff;--c-primary:#0d9488;--c-primary-hover:#0f766e;
            --c-accent:#6366f1;--c-success:#22c55e;--c-text:#0f172a;--c-text-muted:#64748b;
            --c-border:#e2e8f0;--c-row-alt:#f8fafc;--c-highlight:#f0fdfa;
            --radius:10px;--shadow:0 1px 3px rgba(0,0,0,.06),0 1px 2px rgba(0,0,0,.04);
            --shadow-lg:0 4px 16px rgba(0,0,0,.08);--transition:.2s ease;
        }
        *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
        body{font-family:'Inter',system-ui,sans-serif;background:var(--c-bg);color:var(--c-text);padding:2rem;font-size:14px;line-height:1.6;-webkit-font-smoothing:antialiased}
        .report-header{background:linear-gradient(135deg,#0d9488,#6366f1);color:#fff;padding:2rem 2.5rem;border-radius:var(--radius);margin-bottom:2rem;box-shadow:var(--shadow-lg);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem}
        .report-header h1{font-size:1.75rem;font-weight:700;letter-spacing:-.02em}
        .report-header .subtitle{opacity:.85;font-size:.85rem}
        .stats-bar{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:1rem;margin-bottom:2rem}
        .stat-card{background:var(--c-surface);border-radius:var(--radius);padding:1.2rem 1.4rem;box-shadow:var(--shadow);border-left:4px solid var(--c-primary);transition:transform var(--transition),box-shadow var(--transition)}
        .stat-card:hover{transform:translateY(-2px);box-shadow:var(--shadow-lg)}
        .stat-card .label{font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;color:var(--c-text-muted);margin-bottom:.2rem}
        .stat-card .value{font-size:1.45rem;font-weight:700;color:var(--c-primary)}
        .tabs-wrapper{background:var(--c-surface);border-radius:var(--radius);box-shadow:var(--shadow);overflow:hidden}
        .tab-bar{display:flex;border-bottom:2px solid var(--c-border);background:var(--c-row-alt);overflow-x:auto}
        .tab-bar button{flex:1;min-width:max-content;padding:1rem 1.25rem;border:none;background:transparent;font:inherit;font-weight:600;font-size:.88rem;color:var(--c-text-muted);cursor:pointer;position:relative;transition:color var(--transition),background var(--transition);white-space:nowrap}
        .tab-bar button:hover{background:var(--c-highlight);color:var(--c-primary)}
        .tab-bar button.active{color:var(--c-primary);background:var(--c-surface)}
        .tab-bar button.active::after{content:'';position:absolute;bottom:-2px;left:0;right:0;height:3px;background:var(--c-primary);border-radius:3px 3px 0 0}
        .tab-panel{display:none;padding:2rem;animation:fadeIn .25s ease}
        .tab-panel.active{display:block}
        @keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
        .section-title{font-size:1.15rem;font-weight:700;margin-bottom:1.2rem;display:flex;align-items:center;gap:.5rem}
        .section-title .icon{font-size:1.25rem}
        .subsection-label{font-size:.82rem;font-weight:600;color:var(--c-text-muted);text-transform:uppercase;letter-spacing:.04em;margin:1.5rem 0 .75rem;padding-bottom:.4rem;border-bottom:1px dashed var(--c-border)}
        table{width:100%;border-collapse:collapse;font-size:.875rem}
        th,td{padding:.7rem 1rem;text-align:left;border-bottom:1px solid var(--c-border);vertical-align:top}
        th{background:var(--c-row-alt);font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.04em;color:var(--c-text-muted);position:sticky;top:0}
        tbody tr:hover{background:var(--c-highlight)}
        .nested-table{font-size:.8rem;margin-top:.5rem;border:1px solid var(--c-border);border-radius:6px;overflow:hidden}
        .nested-table th{background:#f0fdf4}
        .step-accordion{margin-bottom:.75rem;border:1px solid var(--c-border);border-radius:var(--radius);overflow:hidden;transition:box-shadow var(--transition)}
        .step-accordion:hover{box-shadow:var(--shadow)}
        .step-accordion summary{display:flex;align-items:center;gap:.75rem;padding:1rem 1.25rem;font-weight:600;font-size:.95rem;color:var(--c-text);cursor:pointer;list-style:none;background:var(--c-row-alt);transition:background var(--transition)}
        .step-accordion summary::-webkit-details-marker{display:none}
        .step-accordion summary:hover{background:var(--c-highlight)}
        .step-accordion[open] summary{background:var(--c-highlight);border-bottom:1px solid var(--c-border)}
        .step-number{display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:50%;background:var(--c-primary);color:#fff;font-size:.8rem;font-weight:700;flex-shrink:0}
        .chevron{margin-left:auto;transition:transform .2s;color:var(--c-text-muted);font-size:.8rem}
        .step-accordion[open] .chevron{transform:rotate(90deg)}
        .step-body{padding:1.25rem 1.5rem}
        .step-desc{color:var(--c-text-muted);margin-bottom:1rem;font-size:.875rem}
        .sub-section-title{font-weight:600;font-size:.85rem;color:var(--c-primary-hover);margin:1rem 0 .5rem;display:flex;align-items:center;gap:.35rem}
        .badge{display:inline-block;padding:.15rem .55rem;border-radius:999px;font-size:.7rem;font-weight:600;background:#dbeafe;color:#1e40af;margin:1px 2px}
        .charts-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:1.5rem}
        .chart-card{background:var(--c-surface);border:1px solid var(--c-border);border-radius:var(--radius);overflow:hidden;transition:box-shadow var(--transition)}
        .chart-card:hover{box-shadow:var(--shadow-lg)}
        .chart-header{display:flex;justify-content:space-between;align-items:center;padding:.85rem 1.1rem;border-bottom:1px solid var(--c-border);background:var(--c-row-alt);flex-wrap:wrap;gap:.5rem}
        .chart-header h4{font-size:.9rem;font-weight:600;color:var(--c-text)}
        .chart-header .tag{font-size:.65rem;padding:.15rem .45rem;border-radius:4px;font-weight:600;margin-left:.4rem}
        .tag-onehot{background:#fef3c7;color:#92400e}
        .tag-numeric{background:#dbeafe;color:#1e40af}
        .tag-categorical{background:#ede9fe;color:#5b21b6}
        .chart-toggle{display:flex;gap:0;border:1px solid var(--c-border);border-radius:6px;overflow:hidden}
        .chart-toggle button{padding:.3rem .65rem;border:none;background:transparent;font:inherit;font-size:.7rem;font-weight:600;color:var(--c-text-muted);cursor:pointer;transition:all var(--transition)}
        .chart-toggle button.active{background:var(--c-primary);color:#fff}
        .chart-toggle button:hover:not(.active){background:var(--c-highlight)}
        .chart-body{padding:1rem;min-height:220px;display:flex;align-items:center;justify-content:center}
        .chart-body canvas{max-width:100%;height:200px}
        .chart-stats{display:flex;gap:1rem;padding:.6rem 1.1rem;border-top:1px solid var(--c-border);background:var(--c-row-alt);flex-wrap:wrap}
        .chart-stat{font-size:.72rem;color:var(--c-text-muted)}
        .chart-stat strong{color:var(--c-text);font-weight:600}
        .heatmap-wrapper{grid-column:1/-1}
        .heatmap-wrapper .chart-body{padding:1.5rem;overflow-x:auto;justify-content:flex-start}
        .heatmap-wrapper canvas{height:auto!important;max-height:500px}
        .heatmap-legend{display:flex;align-items:center;gap:.5rem;padding:.5rem 1.1rem;border-top:1px solid var(--c-border);background:var(--c-row-alt);font-size:.72rem;color:var(--c-text-muted);flex-wrap:wrap}
        .heatmap-legend .grad{width:120px;height:12px;border-radius:3px;border:1px solid var(--c-border)}
        .report-footer{text-align:center;color:var(--c-text-muted);font-size:.75rem;margin-top:2.5rem;padding-top:1.5rem;border-top:1px solid var(--c-border)}
        @media(max-width:640px){
            body{padding:1rem;font-size:13px}
            .report-header{padding:1.5rem}.report-header h1{font-size:1.3rem}
            .tab-bar button{padding:.75rem .5rem;font-size:.78rem}
            .tab-panel{padding:1.25rem}
            th,td{padding:.5rem .6rem}
            .charts-grid{grid-template-columns:1fr}
        }
    </style>
    """

    # ── JS ──
    js = f"""
    <script>
    function openTab(evt,id){{
        document.querySelectorAll('.tab-panel').forEach(p=>{{p.style.display='none';p.classList.remove('active')}});
        document.querySelectorAll('.tab-bar button').forEach(b=>b.classList.remove('active'));
        const panel=document.getElementById(id);
        panel.style.display='block';void panel.offsetWidth;panel.classList.add('active');
        evt.currentTarget.classList.add('active');
    }}
    window.addEventListener('DOMContentLoaded',()=>document.querySelector('.tab-bar button').click());

    const COL_DATA = {col_values_json};
    const NUM_COLS = {numeric_cols_json};
    const CAT_COLS = {categorical_cols_json};
    const CORR_COLS = {corr_cols_json};
    const OH_GROUPS = {onehot_groups_json};

    const PAL = {{
        primary:'#0d9488',accent:'#6366f1',success:'#22c55e',warm:'#f59e0b',rose:'#f43f5e',
        grid:'#e2e8f0',text:'#475569',textLight:'#94a3b8',
        fills:['rgba(13,148,136,.2)','rgba(99,102,241,.2)','rgba(34,197,94,.2)','rgba(245,158,11,.2)','rgba(244,63,94,.2)','rgba(168,85,247,.2)','rgba(14,165,233,.2)','rgba(251,146,60,.2)'],
        strokes:['#0d9488','#6366f1','#22c55e','#f59e0b','#f43f5e','#a855f7','#0ea5e9','#fb923c'],
        gradientFills:['rgba(13,148,136,.35)','rgba(99,102,241,.35)','rgba(34,197,94,.35)'],
        heatmap: n => {{
            const t=Math.max(0,Math.min(1,(n+1)/2));
            if(t<0.5){{const s=t*2;return `rgb(${{Math.round(59+s*(255-59))}},${{Math.round(130+s*(255-130))}},${{Math.round(246+s*(255-246))}})`;}}
            else{{const s=(t-0.5)*2;return `rgb(255,${{Math.round(255-s*(255-87))}},${{Math.round(255-s*(255-87))}})`;}}
        }}
    }};

    function stats(arr){{
        const s=[...arr].sort((a,b)=>a-b),n=s.length;
        const sum=s.reduce((a,b)=>a+b,0),mean=sum/n;
        const q=p=>{{const i=p*(n-1),lo=Math.floor(i);return lo===i?s[lo]:s[lo]+(s[lo+1]-s[lo])*(i-lo);}};
        const variance=s.reduce((a,v)=>a+(v-mean)**2,0)/n;
        return {{min:s[0],max:s[n-1],mean,median:q(.5),q1:q(.25),q3:q(.75),std:Math.sqrt(variance),n}};
    }}

    function setupCanvas(canvas,w,h){{
        const dpr=window.devicePixelRatio||1;
        canvas.width=w*dpr;canvas.height=h*dpr;
        canvas.style.width=w+'px';canvas.style.height=h+'px';
        const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);return ctx;
    }}

    function fmtTick(v,range){{
        if(Math.abs(v)>=1e6) return (v/1e6).toFixed(1)+'M';
        if(Math.abs(v)>=1e3) return (v/1e3).toFixed(1)+'K';
        if(range<1) return v.toFixed(3);
        if(range<10) return v.toFixed(2);
        if(range<100) return v.toFixed(1);
        return Math.round(v).toString();
    }}
    function fmtRange(v){{
        if(Math.abs(v)>=1e6) return (v/1e6).toFixed(1)+'M';
        if(Math.abs(v)>=1e3) return (v/1e3).toFixed(1)+'K';
        return Number.isInteger(v)?v.toString():v.toFixed(1);
    }}

    function niceScale(dMin,dMax,maxTicks){{
        const range=dMax-dMin||1;const rough=range/maxTicks;
        const mag=Math.pow(10,Math.floor(Math.log10(rough)));const residual=rough/mag;
        let nice;
        if(residual<=1.5)nice=mag;else if(residual<=3)nice=2*mag;else if(residual<=7)nice=5*mag;else nice=10*mag;
        const lo=Math.floor(dMin/nice)*nice,hi=Math.ceil(dMax/nice)*nice;
        const ticks=[];for(let v=lo;v<=hi+nice*0.001;v+=nice)ticks.push(parseFloat(v.toFixed(10)));
        return {{lo,hi,ticks}};
    }}

    function drawAxes(ctx,m,w,h,yMin,yMax){{
        const range=yMax-yMin;const ns=niceScale(yMin,yMax,5);
        ctx.strokeStyle=PAL.grid;ctx.lineWidth=1;
        ctx.beginPath();ctx.moveTo(m.l,m.t);ctx.lineTo(m.l,h-m.b);ctx.lineTo(w-m.r,h-m.b);ctx.stroke();
        ctx.fillStyle=PAL.textLight;ctx.font='10px Inter,sans-serif';ctx.textAlign='right';
        ns.ticks.forEach(v=>{{
            const y=m.t+(h-m.t-m.b)*(1-(v-ns.lo)/(ns.hi-ns.lo||1));
            if(y<m.t-2||y>h-m.b+2)return;
            ctx.fillText(fmtTick(v,range),m.l-6,y+3);
            ctx.strokeStyle='#f1f5f9';ctx.beginPath();ctx.moveTo(m.l+1,y);ctx.lineTo(w-m.r,y);ctx.stroke();ctx.strokeStyle=PAL.grid;
        }});
        return ns;
    }}

    function drawXLabels(ctx,m,w,h,xMin,xMax,numTicks){{
        const range=xMax-xMin;const ns=niceScale(xMin,xMax,numTicks||6);
        ctx.fillStyle=PAL.textLight;ctx.font='9px Inter,sans-serif';ctx.textAlign='center';
        ns.ticks.forEach(v=>{{
            const x=m.l+(w-m.l-m.r)*((v-ns.lo)/(ns.hi-ns.lo||1));
            if(x<m.l-2||x>w-m.r+2)return;
            ctx.fillText(fmtTick(v,range),x,h-m.b+14);
        }});
        return ns;
    }}

    // ── Histogram ──
    function drawHistogram(canvas,data,colName){{
        const W=canvas.parentElement.clientWidth-32||380,H=200;
        const ctx=setupCanvas(canvas,W,H);
        const s=stats(data),m={{t:15,r:15,b:30,l:48}};
        const numBins=Math.min(Math.ceil(Math.sqrt(data.length)),30);
        const binW=(s.max-s.min)/numBins||1;
        const bins=Array(numBins).fill(0);
        data.forEach(v=>{{let i=Math.floor((v-s.min)/binW);if(i>=numBins)i=numBins-1;bins[i]++;}});
        const maxBin=Math.max(...bins);
        const ns=drawAxes(ctx,m,W,H,0,maxBin);
        const xNs=drawXLabels(ctx,m,W,H,s.min,s.max,6);
        const plotW=W-m.l-m.r;
        const xScale=v=>(v-xNs.lo)/(xNs.hi-xNs.lo||1)*plotW;
        const cIdx=NUM_COLS.indexOf(colName)%PAL.fills.length;
        ctx.fillStyle=PAL.fills[cIdx];ctx.strokeStyle=PAL.strokes[cIdx];ctx.lineWidth=1.5;
        bins.forEach((c,i)=>{{
            const x1=m.l+xScale(s.min+i*binW),x2=m.l+xScale(s.min+(i+1)*binW);
            const bh=ns.hi?c/ns.hi*(H-m.t-m.b):0,y=H-m.b-bh;
            ctx.beginPath();ctx.rect(x1+.5,y,x2-x1-1,bh);ctx.fill();ctx.stroke();
        }});
        const mx=m.l+xScale(s.mean);
        ctx.strokeStyle=PAL.rose;ctx.lineWidth=1.5;ctx.setLineDash([5,3]);
        ctx.beginPath();ctx.moveTo(mx,m.t);ctx.lineTo(mx,H-m.b);ctx.stroke();ctx.setLineDash([]);
        ctx.fillStyle=PAL.rose;ctx.font='9px Inter,sans-serif';ctx.textAlign='left';ctx.fillText('μ',mx+3,m.t+10);
    }}

    // ── Box Plot ──
    function drawBoxPlot(canvas,data,colName){{
        const W=canvas.parentElement.clientWidth-32||380,H=200;
        const ctx=setupCanvas(canvas,W,H);
        const s=stats(data),m={{t:20,r:20,b:25,l:48}};
        const plotW=W-m.l-m.r,plotH=H-m.t-m.b;
        const cIdx=NUM_COLS.indexOf(colName)%PAL.fills.length;
        const iqr=s.q3-s.q1;const wLo=Math.max(s.min,s.q1-1.5*iqr),wHi=Math.min(s.max,s.q3+1.5*iqr);
        const outliers=data.filter(v=>v<wLo||v>wHi);
        const dMin=Math.min(wLo,...outliers),dMax=Math.max(wHi,...outliers);
        drawAxes(ctx,m,W,H,dMin,dMax);
        const mapY=v=>m.t+plotH*(1-(v-dMin)/(dMax-dMin||1));
        const cx=m.l+plotW/2,bw=Math.min(plotW*.4,70);
        ctx.strokeStyle=PAL.strokes[cIdx];ctx.lineWidth=1.5;
        ctx.setLineDash([4,3]);ctx.beginPath();ctx.moveTo(cx,mapY(wLo));ctx.lineTo(cx,mapY(s.q1));ctx.moveTo(cx,mapY(s.q3));ctx.lineTo(cx,mapY(wHi));ctx.stroke();ctx.setLineDash([]);
        ctx.beginPath();ctx.moveTo(cx-bw*.3,mapY(wLo));ctx.lineTo(cx+bw*.3,mapY(wLo));ctx.moveTo(cx-bw*.3,mapY(wHi));ctx.lineTo(cx+bw*.3,mapY(wHi));ctx.stroke();
        const bTop=mapY(s.q3),bBot=mapY(s.q1);
        ctx.fillStyle=PAL.fills[cIdx];ctx.beginPath();ctx.rect(cx-bw/2,bTop,bw,bBot-bTop);ctx.fill();ctx.stroke();
        ctx.strokeStyle=PAL.primary;ctx.lineWidth=2.5;ctx.beginPath();ctx.moveTo(cx-bw/2,mapY(s.median));ctx.lineTo(cx+bw/2,mapY(s.median));ctx.stroke();
        const my=mapY(s.mean);ctx.fillStyle=PAL.accent;ctx.beginPath();ctx.moveTo(cx,my-5);ctx.lineTo(cx+5,my);ctx.lineTo(cx,my+5);ctx.lineTo(cx-5,my);ctx.closePath();ctx.fill();
        ctx.fillStyle='rgba(244,63,94,.5)';ctx.strokeStyle=PAL.rose;ctx.lineWidth=1;
        outliers.forEach(v=>{{const y=mapY(v);ctx.beginPath();ctx.arc(cx+(Math.random()-.5)*bw*.4,y,3,0,Math.PI*2);ctx.fill();ctx.stroke();}});
    }}

    // ── Violin Plot ──
    function drawViolin(canvas,data,colName){{
        const W=canvas.parentElement.clientWidth-32||380,H=200;
        const ctx=setupCanvas(canvas,W,H);
        const s=stats(data),m={{t:20,r:20,b:25,l:48}};
        const plotH=H-m.t-m.b,cx=(W-m.l-m.r)/2+m.l;
        const cIdx=NUM_COLS.indexOf(colName)%PAL.fills.length;
        drawAxes(ctx,m,W,H,s.min,s.max);
        const mapY=v=>m.t+plotH*(1-(v-s.min)/(s.max-s.min||1));
        const bw=(s.q3-s.q1)*0.6/(data.length**0.2)||1;
        const steps=60,kde=[];let maxD=0;
        for(let i=0;i<=steps;i++){{
            const v=s.min+(s.max-s.min)*i/steps;let d=0;
            data.forEach(x=>{{const z=(v-x)/bw;d+=Math.exp(-.5*z*z);}});
            d/=data.length*bw*Math.sqrt(2*Math.PI);kde.push({{v,d}});if(d>maxD)maxD=d;
        }}
        const maxW=Math.min((W-m.l-m.r)*.35,80);
        ctx.fillStyle=PAL.fills[cIdx];ctx.strokeStyle=PAL.strokes[cIdx];ctx.lineWidth=1.5;
        ctx.beginPath();
        kde.forEach((p,i)=>{{const y=mapY(p.v),w=maxD?p.d/maxD*maxW:0;if(i===0)ctx.moveTo(cx-w,y);else ctx.lineTo(cx-w,y);}});
        for(let i=kde.length-1;i>=0;i--){{const p=kde[i],y=mapY(p.v),w=maxD?p.d/maxD*maxW:0;ctx.lineTo(cx+w,y);}}
        ctx.closePath();ctx.fill();ctx.stroke();
        ctx.strokeStyle=PAL.primary;ctx.lineWidth=2;
        [s.q1,s.median,s.q3].forEach((v,i)=>{{
            const y=mapY(v),idx=Math.round((v-s.min)/(s.max-s.min||1)*steps);
            const d=kde[Math.min(idx,kde.length-1)]?.d||0,w=maxD?d/maxD*maxW:0;
            ctx.setLineDash(i===1?[]:[4,3]);ctx.beginPath();ctx.moveTo(cx-w,y);ctx.lineTo(cx+w,y);ctx.stroke();
        }});ctx.setLineDash([]);
        ctx.fillStyle='rgba(99,102,241,.25)';
        data.forEach(v=>{{
            const y=mapY(v),idx=Math.round((v-s.min)/(s.max-s.min||1)*steps);
            const d=kde[Math.min(idx,kde.length-1)]?.d||0,w=maxD?d/maxD*maxW*.7:0;
            ctx.beginPath();ctx.arc(cx+(Math.random()-.5)*2*w,y,1.5,0,Math.PI*2);ctx.fill();
        }});
    }}

    // ── Bar Chart (Categorical) ──
    function drawBarChart(canvas,data,colName){{
        const W=canvas.parentElement.clientWidth-32||380,H=200;
        const ctx=setupCanvas(canvas,W,H);
        const counts={{}};data.forEach(v=>{{const k=String(v);counts[k]=(counts[k]||0)+1;}});
        const entries=Object.entries(counts).sort((a,b)=>b[1]-a[1]).slice(0,12);
        const maxC=Math.max(...entries.map(e=>e[1]));
        const m={{t:15,r:15,b:50,l:48}};const plotW=W-m.l-m.r,plotH=H-m.t-m.b;
        const bw=plotW/entries.length;const ns=drawAxes(ctx,m,W,H,0,maxC);
        const cIdx=CAT_COLS.indexOf(colName);
        entries.forEach(([label,count],i)=>{{
            const bh=ns.hi?count/ns.hi*plotH:0;const x=m.l+i*bw,y=H-m.b-bh;
            ctx.fillStyle=PAL.fills[(cIdx+i)%PAL.fills.length];ctx.strokeStyle=PAL.strokes[(cIdx+i)%PAL.strokes.length];ctx.lineWidth=1.5;
            ctx.beginPath();ctx.rect(x+2,y,bw-4,bh);ctx.fill();ctx.stroke();
            if(bh>18){{ctx.fillStyle=PAL.text;ctx.font='bold 10px Inter,sans-serif';ctx.textAlign='center';ctx.fillText(count,x+bw/2,y+14);}}
            ctx.fillStyle=PAL.textLight;ctx.font='9px Inter,sans-serif';ctx.textAlign='right';
            ctx.save();ctx.translate(x+bw/2,H-m.b+6);ctx.rotate(-Math.PI/4);
            ctx.fillText(label.length>12?label.slice(0,11)+'…':label,0,0);ctx.restore();
        }});
    }}

    // ── Donut ──
    function drawDonut(canvas,data,colName){{
        const W=canvas.parentElement.clientWidth-32||380,H=220;
        const ctx=setupCanvas(canvas,W,H);
        const counts={{}};data.forEach(v=>{{const k=String(v);counts[k]=(counts[k]||0)+1;}});
        const entries=Object.entries(counts).sort((a,b)=>b[1]-a[1]);
        const total=data.length;const topN=8;
        let shown=entries.slice(0,topN);const otherC=entries.slice(topN).reduce((s,e)=>s+e[1],0);
        if(otherC>0)shown.push(['Other',otherC]);
        const cx=W/2-60,cy=H/2,r=Math.min(cx-10,cy-10,80),inner=r*0.55;
        let angle=-Math.PI/2;const cIdx=CAT_COLS.indexOf(colName);const legendX=cx+r+25;
        shown.forEach(([label,count],i)=>{{
            const sweep=count/total*Math.PI*2;
            ctx.fillStyle=PAL.strokes[(cIdx+i)%PAL.strokes.length];
            ctx.beginPath();ctx.moveTo(cx+Math.cos(angle)*inner,cy+Math.sin(angle)*inner);
            ctx.arc(cx,cy,r,angle,angle+sweep);ctx.arc(cx,cy,inner,angle+sweep,angle,true);ctx.closePath();ctx.fill();
            const ly=20+i*18;ctx.fillRect(legendX,ly,10,10);
            ctx.fillStyle=PAL.text;ctx.font='10px Inter,sans-serif';ctx.textAlign='left';
            ctx.fillText(`${{(label.length>14?label.slice(0,13)+'…':label)}} (${{(count/total*100).toFixed(1)}}%)`,legendX+14,ly+9);
            ctx.fillStyle=PAL.strokes[(cIdx+i)%PAL.strokes.length];
            angle+=sweep;
        }});
        ctx.fillStyle=PAL.text;ctx.font='bold 16px Inter,sans-serif';ctx.textAlign='center';ctx.fillText(total.toString(),cx,cy+2);
        ctx.fillStyle=PAL.textLight;ctx.font='9px Inter,sans-serif';ctx.fillText('total',cx,cy+14);
    }}

    // ── Pareto ──
    function drawPareto(canvas,data,colName){{
        const W=canvas.parentElement.clientWidth-32||380,H=200;
        const ctx=setupCanvas(canvas,W,H);
        const counts={{}};data.forEach(v=>{{const k=String(v);counts[k]=(counts[k]||0)+1;}});
        const entries=Object.entries(counts).sort((a,b)=>b[1]-a[1]).slice(0,12);
        const maxC=entries[0][1];const total=entries.reduce((s,e)=>s+e[1],0);
        const m={{t:15,r:48,b:50,l:48}};const plotW=W-m.l-m.r,plotH=H-m.t-m.b;
        const bw=plotW/entries.length;const ns=drawAxes(ctx,m,W,H,0,maxC);
        ctx.strokeStyle=PAL.grid;ctx.beginPath();ctx.moveTo(W-m.r,m.t);ctx.lineTo(W-m.r,H-m.b);ctx.stroke();
        ctx.fillStyle=PAL.rose;ctx.font='10px Inter,sans-serif';ctx.textAlign='left';
        [0,25,50,75,100].forEach(p=>{{ctx.fillText(p+'%',W-m.r+5,m.t+plotH*(1-p/100)+3);}});
        const cIdx=CAT_COLS.indexOf(colName);let cumul=0;const cumulPts=[];
        entries.forEach(([label,count],i)=>{{
            const bh=ns.hi?count/ns.hi*plotH:0;const x=m.l+i*bw,y=H-m.b-bh;
            ctx.fillStyle=PAL.fills[(cIdx+i)%PAL.fills.length];ctx.strokeStyle=PAL.strokes[(cIdx+i)%PAL.strokes.length];ctx.lineWidth=1.5;
            ctx.beginPath();ctx.rect(x+2,y,bw-4,bh);ctx.fill();ctx.stroke();
            cumul+=count;cumulPts.push({{x:x+bw/2,y:m.t+plotH*(1-cumul/total)}});
            ctx.fillStyle=PAL.textLight;ctx.font='9px Inter,sans-serif';ctx.textAlign='right';
            ctx.save();ctx.translate(x+bw/2,H-m.b+6);ctx.rotate(-Math.PI/4);
            ctx.fillText(label.length>12?label.slice(0,11)+'…':label,0,0);ctx.restore();
        }});
        ctx.strokeStyle=PAL.rose;ctx.lineWidth=2;ctx.beginPath();
        cumulPts.forEach((p,i)=>{{if(i===0)ctx.moveTo(p.x,p.y);else ctx.lineTo(p.x,p.y);}});ctx.stroke();
        cumulPts.forEach(p=>{{ctx.fillStyle='#fff';ctx.strokeStyle=PAL.rose;ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(p.x,p.y,3.5,0,Math.PI*2);ctx.fill();ctx.stroke();}});
    }}

    // ══════════════════════════════════════════════════════════════
    // ONE-HOT RANGE CHARTS
    // ══════════════════════════════════════════════════════════════

    function getOHData(safeId){{
        const group=OH_GROUPS[safeId];
        if(!group||!group.bins)return[];
        return group.bins.map(b=>{{
            const vals=COL_DATA[b.col]||[];
            const count=vals.filter(v=>v===1||v===1.0||v===true||v==='1').length;
            return {{col:b.col, lo:b.lo, hi:b.hi, count, label:fmtRange(b.lo)+'-'+fmtRange(b.hi)}};
        }});
    }}

    // ── Range Bar (ordered) ──
    function drawRangeBar(canvas,safeId){{
        const bins=getOHData(safeId);if(!bins.length)return;
        const W=canvas.parentElement.clientWidth-32||380,H=200;
        const ctx=setupCanvas(canvas,W,H);
        const maxC=Math.max(...bins.map(b=>b.count));
        const m={{t:15,r:15,b:52,l:48}};const plotW=W-m.l-m.r,plotH=H-m.t-m.b;
        const bw=plotW/bins.length;
        const ns=drawAxes(ctx,m,W,H,0,maxC);
        const total=bins.reduce((s,b)=>s+b.count,0);
        bins.forEach((b,i)=>{{
            const bh=ns.hi?b.count/ns.hi*plotH:0;const x=m.l+i*bw,y=H-m.b-bh;
            // gradient fill
            const grad=ctx.createLinearGradient(x,y,x,H-m.b);
            grad.addColorStop(0,PAL.strokes[i%PAL.strokes.length]);
            grad.addColorStop(1,PAL.fills[i%PAL.fills.length]);
            ctx.fillStyle=grad;ctx.strokeStyle=PAL.strokes[i%PAL.strokes.length];ctx.lineWidth=1.5;
            ctx.beginPath();ctx.rect(x+2,y,bw-4,bh);ctx.fill();ctx.stroke();
            // pct label on bar
            if(bh>22){{
                const pct=total?(b.count/total*100).toFixed(1)+'%':'';
                ctx.fillStyle='#fff';ctx.font='bold 10px Inter,sans-serif';ctx.textAlign='center';ctx.fillText(pct,x+bw/2,y+14);
            }}
            // count below bar
            ctx.fillStyle=PAL.text;ctx.font='9px Inter,sans-serif';ctx.textAlign='center';ctx.fillText(b.count,x+bw/2,y-4);
            // range label
            ctx.fillStyle=PAL.textLight;ctx.font='9px Inter,sans-serif';ctx.textAlign='right';
            ctx.save();ctx.translate(x+bw/2,H-m.b+6);ctx.rotate(-Math.PI/4);
            ctx.fillText(b.label,0,0);ctx.restore();
        }});
    }}

    // ── Cumulative Step ──
    function drawCumulStep(canvas,safeId){{
        const bins=getOHData(safeId);if(!bins.length)return;
        const W=canvas.parentElement.clientWidth-32||380,H=200;
        const ctx=setupCanvas(canvas,W,H);
        const total=bins.reduce((s,b)=>s+b.count,0);
        const m={{t:20,r:48,b:30,l:48}};const plotW=W-m.l-m.r,plotH=H-m.t-m.b;
        // y axis: count
        const ns=drawAxes(ctx,m,W,H,0,total);
        // right axis: percentage
        ctx.strokeStyle=PAL.grid;ctx.beginPath();ctx.moveTo(W-m.r,m.t);ctx.lineTo(W-m.r,H-m.b);ctx.stroke();
        ctx.fillStyle=PAL.accent;ctx.font='10px Inter,sans-serif';ctx.textAlign='left';
        [0,25,50,75,100].forEach(p=>{{ctx.fillText(p+'%',W-m.r+5,m.t+plotH*(1-p/100)+3);}});
        // x labels
        const xStep=plotW/(bins.length);
        ctx.fillStyle=PAL.textLight;ctx.font='9px Inter,sans-serif';ctx.textAlign='center';
        bins.forEach((b,i)=>ctx.fillText(b.label,m.l+i*xStep+xStep/2,H-m.b+14));

        let cumul=0;const pts=[];
        bins.forEach((b,i)=>{{
            cumul+=b.count;
            pts.push({{x:m.l+i*xStep+xStep/2, y:m.t+plotH*(1-cumul/(ns.hi||1))}});
        }});
        // area fill
        ctx.beginPath();ctx.moveTo(pts[0].x,H-m.b);
        pts.forEach((p,i)=>{{
            if(i>0)ctx.lineTo(p.x,pts[i-1].y); // horizontal step
            ctx.lineTo(p.x,p.y);
        }});
        ctx.lineTo(pts[pts.length-1].x,H-m.b);ctx.closePath();
        const grad=ctx.createLinearGradient(0,m.t,0,H-m.b);
        grad.addColorStop(0,'rgba(13,148,136,.3)');grad.addColorStop(1,'rgba(13,148,136,.02)');
        ctx.fillStyle=grad;ctx.fill();
        // step line
        ctx.strokeStyle=PAL.primary;ctx.lineWidth=2.5;ctx.beginPath();
        pts.forEach((p,i)=>{{
            if(i===0)ctx.moveTo(p.x,p.y);
            else{{ctx.lineTo(p.x,pts[i-1].y);ctx.lineTo(p.x,p.y);}}
        }});ctx.stroke();
        // dots
        pts.forEach((p,i)=>{{
            ctx.fillStyle='#fff';ctx.strokeStyle=PAL.primary;ctx.lineWidth=2;
            ctx.beginPath();ctx.arc(p.x,p.y,4,0,Math.PI*2);ctx.fill();ctx.stroke();
            const pct=(bins.slice(0,i+1).reduce((s,b)=>s+b.count,0)/total*100).toFixed(0);
            ctx.fillStyle=PAL.text;ctx.font='bold 9px Inter,sans-serif';ctx.textAlign='center';
            ctx.fillText(pct+'%',p.x,p.y-8);
        }});
    }}

    // ── Proportion / Stacked Percentage Bar ──
    function drawProportion(canvas,safeId){{
        const bins=getOHData(safeId);if(!bins.length)return;
        const W=canvas.parentElement.clientWidth-32||380,H=200;
        const ctx=setupCanvas(canvas,W,H);
        const total=bins.reduce((s,b)=>s+b.count,0);
        const m={{t:25,r:15,b:70,l:15}};const barH=50;const barY=(H-m.b-m.t)/2-barH/2+m.t;
        // stacked horizontal bar
        let x=m.l;
        const barW=W-m.l-m.r;
        bins.forEach((b,i)=>{{
            const w=total?b.count/total*barW:0;
            ctx.fillStyle=PAL.strokes[i%PAL.strokes.length];
            ctx.beginPath();
            if(i===0&&bins.length>1){{
                ctx.moveTo(x+6,barY);ctx.lineTo(x+w,barY);ctx.lineTo(x+w,barY+barH);ctx.lineTo(x+6,barY+barH);
                ctx.quadraticCurveTo(x,barY+barH,x,barY+barH-6);ctx.lineTo(x,barY+6);ctx.quadraticCurveTo(x,barY,x+6,barY);
            }} else if(i===bins.length-1){{
                ctx.moveTo(x,barY);ctx.lineTo(x+w-6,barY);ctx.quadraticCurveTo(x+w,barY,x+w,barY+6);
                ctx.lineTo(x+w,barY+barH-6);ctx.quadraticCurveTo(x+w,barY+barH,x+w-6,barY+barH);ctx.lineTo(x,barY+barH);
            }} else {{
                ctx.rect(x,barY,w,barH);
            }}
            ctx.fill();
            // pct label inside
            const pct=(b.count/total*100);
            if(w>30){{
                ctx.fillStyle='#fff';ctx.font='bold 11px Inter,sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';
                ctx.fillText(pct.toFixed(1)+'%',x+w/2,barY+barH/2);
            }}
            // label below
            ctx.fillStyle=PAL.strokes[i%PAL.strokes.length];ctx.font='bold 9px Inter,sans-serif';ctx.textAlign='center';
            const lblX=x+w/2;
            // connector line
            ctx.strokeStyle=PAL.strokes[i%PAL.strokes.length];ctx.lineWidth=1;ctx.setLineDash([2,2]);
            ctx.beginPath();ctx.moveTo(lblX,barY+barH);ctx.lineTo(lblX,barY+barH+14);ctx.stroke();ctx.setLineDash([]);
            ctx.save();ctx.translate(lblX,barY+barH+18);ctx.rotate(-Math.PI/5);
            ctx.fillStyle=PAL.text;ctx.font='9px Inter,sans-serif';ctx.textAlign='right';
            ctx.fillText(b.label,0,0);ctx.restore();
            // count
            ctx.fillStyle=PAL.textLight;ctx.font='8px Inter,sans-serif';ctx.textAlign='center';
            ctx.fillText('n='+b.count,lblX,barY-6);
            x+=w;
        }});
        // title
        ctx.fillStyle=PAL.text;ctx.font='bold 10px Inter,sans-serif';ctx.textAlign='left';
        ctx.fillText('Distribution across bins (n='+total+')',m.l,m.t-8);
    }}

    // ══════════════════════════════════════════════════════════════
    // CORRELATION
    // ══════════════════════════════════════════════════════════════

    function rankArray(arr){{
        const indexed=arr.map((v,i)=>({{v,i}})).sort((a,b)=>a.v-b.v);
        const ranks=new Array(arr.length);let i=0;
        while(i<indexed.length){{
            let j=i;while(j<indexed.length&&indexed[j].v===indexed[i].v)j++;
            const avg=(i+j-1)/2+1;for(let k=i;k<j;k++)ranks[indexed[k].i]=avg;i=j;
        }}
        return ranks;
    }}
    function pearsonCorr(a,b){{
        const paired=[];for(let i=0;i<Math.min(a.length,b.length);i++)if(typeof a[i]==='number'&&typeof b[i]==='number')paired.push([a[i],b[i]]);
        if(paired.length<3)return 0;
        const ma=paired.reduce((s,p)=>s+p[0],0)/paired.length,mb=paired.reduce((s,p)=>s+p[1],0)/paired.length;
        let num=0,da=0,db=0;paired.forEach(([x,y])=>{{num+=(x-ma)*(y-mb);da+=(x-ma)**2;db+=(y-mb)**2;}});
        return da&&db?num/Math.sqrt(da*db):0;
    }}
    function spearmanCorr(a,b){{
        const paired=[];for(let i=0;i<Math.min(a.length,b.length);i++)if(typeof a[i]==='number'&&typeof b[i]==='number')paired.push([a[i],b[i]]);
        if(paired.length<3)return 0;return pearsonCorr(rankArray(paired.map(p=>p[0])),rankArray(paired.map(p=>p[1])));
    }}
    function kendallCorr(a,b){{
        const paired=[];for(let i=0;i<Math.min(a.length,b.length);i++)if(typeof a[i]==='number'&&typeof b[i]==='number')paired.push([a[i],b[i]]);
        const n=paired.length;if(n<3)return 0;let conc=0,disc=0;
        for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){{const dx=paired[i][0]-paired[j][0],dy=paired[i][1]-paired[j][1];if(dx*dy>0)conc++;else if(dx*dy<0)disc++;}}
        const denom=n*(n-1)/2;return denom?(conc-disc)/denom:0;
    }}

    let currentCorrMethod='pearson';
    function drawHeatmap(canvas,method){{
        if(CORR_COLS.length<2)return;
        method=method||currentCorrMethod;currentCorrMethod=method;
        const corrFn=method==='spearman'?spearmanCorr:method==='kendall'?kendallCorr:pearsonCorr;
        const n=CORR_COLS.length;
        const tmpCtx=canvas.getContext('2d');tmpCtx.font='10px Inter,sans-serif';
        let maxLabelW=0;CORR_COLS.forEach(c=>{{const w=tmpCtx.measureText(c).width;if(w>maxLabelW)maxLabelW=w;}});
        const labelPad=Math.min(maxLabelW+14,150);
        const cellSize=Math.max(28,Math.min(50,Math.floor(480/n)));
        const gridW=n*cellSize,gridH=n*cellSize;
        const W=labelPad+gridW+20,H=labelPad+gridH+20;
        const ctx=setupCanvas(canvas,W,H);const ox=labelPad,oy=labelPad;
        ctx.fillStyle=PAL.text;ctx.font='10px Inter,sans-serif';ctx.textAlign='right';ctx.textBaseline='middle';
        CORR_COLS.forEach((c,i)=>{{ctx.fillText(c.length>18?c.slice(0,17)+'…':c,ox-6,oy+i*cellSize+cellSize/2);}});
        ctx.textAlign='left';ctx.textBaseline='middle';
        CORR_COLS.forEach((c,i)=>{{
            ctx.save();ctx.translate(ox+i*cellSize+cellSize/2,oy-6);ctx.rotate(-Math.PI/3);
            ctx.fillText(c.length>18?c.slice(0,17)+'…':c,0,0);ctx.restore();
        }});
        CORR_COLS.forEach((ca,i)=>{{CORR_COLS.forEach((cb,j)=>{{
            const r=corrFn(COL_DATA[ca],COL_DATA[cb]);const x=ox+j*cellSize,y=oy+i*cellSize;
            ctx.fillStyle=PAL.heatmap(r);ctx.beginPath();
            if(ctx.roundRect)ctx.roundRect(x+.5,y+.5,cellSize-1,cellSize-1,3);else ctx.rect(x+.5,y+.5,cellSize-1,cellSize-1);
            ctx.fill();
            if(cellSize>=30){{ctx.fillStyle=Math.abs(r)>.55?'#fff':'#333';ctx.font='bold 10px Inter,sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(r.toFixed(2),x+cellSize/2,y+cellSize/2);}}
        }});}});
        const card=canvas.closest('.chart-card');
        card.querySelectorAll('.chart-toggle button').forEach(b=>b.classList.toggle('active',b.dataset.type===method));
    }}

    // ── Chart Manager ──
    const chartInstances={{}};
    function renderChart(colName,type){{
        const canvasId='chart-'+colName.replace(/[^a-zA-Z0-9]/g,'_');
        const canvas=document.getElementById(canvasId);if(!canvas)return;
        const ctx=canvas.getContext('2d');const dpr=window.devicePixelRatio||1;
        ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,canvas.width,canvas.height);
        const data=COL_DATA[colName];const nums=data.filter(v=>typeof v==='number');
        if(type==='histogram')drawHistogram(canvas,nums,colName);
        else if(type==='boxplot')drawBoxPlot(canvas,nums,colName);
        else if(type==='violin')drawViolin(canvas,nums,colName);
        else if(type==='bar')drawBarChart(canvas,data,colName);
        else if(type==='donut')drawDonut(canvas,data,colName);
        else if(type==='pareto')drawPareto(canvas,data,colName);
        chartInstances[colName]=type;
        const card=canvas.closest('.chart-card');
        card.querySelectorAll('.chart-toggle button').forEach(b=>b.classList.toggle('active',b.dataset.type===type));
    }}
    function renderOH(safeId,type){{
        const canvas=document.getElementById('oh-'+safeId);if(!canvas)return;
        const ctx=canvas.getContext('2d');const dpr=window.devicePixelRatio||1;
        ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,canvas.width,canvas.height);
        if(type==='rangebar')drawRangeBar(canvas,safeId);
        else if(type==='cumulative')drawCumulStep(canvas,safeId);
        else if(type==='proportion')drawProportion(canvas,safeId);
        chartInstances['oh_'+safeId]=type;
        const card=canvas.closest('.chart-card');
        card.querySelectorAll('.chart-toggle button').forEach(b=>b.classList.toggle('active',b.dataset.type===type));
    }}

    window.addEventListener('DOMContentLoaded',()=>{{
        NUM_COLS.forEach(c=>renderChart(c,'histogram'));
        CAT_COLS.forEach(c=>renderChart(c,'bar'));
        Object.keys(OH_GROUPS).forEach(sid=>renderOH(sid,'rangebar'));
        if(CORR_COLS.length>=2)drawHeatmap(document.getElementById('heatmap-canvas'),'pearson');
    }});
    window.addEventListener('resize',()=>{{
        NUM_COLS.forEach(c=>renderChart(c,chartInstances[c]||'histogram'));
        CAT_COLS.forEach(c=>renderChart(c,chartInstances[c]||'bar'));
        Object.keys(OH_GROUPS).forEach(sid=>renderOH(sid,chartInstances['oh_'+sid]||'rangebar'));
        if(CORR_COLS.length>=2)drawHeatmap(document.getElementById('heatmap-canvas'));
    }});
    </script>
    """

    # ── Python helpers ──
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
    for label, value in [
        ("Pipeline Steps", str(len(steps))),
        ("Initial Rows", str(init_rows) if init_rows is not None else "—"),
        ("Final Rows", str(final_rows) if final_rows is not None else "—"),
        ("Numeric Cols", str(len(numeric_cols))),
        ("Categorical", str(len(categorical_cols))),
        ("One-Hot Groups", str(len(onehot_groups))),
    ]:
        stats_html += f'<div class="stat-card"><div class="label">{label}</div><div class="value">{value}</div></div>'
    stats_html += "</div>"

    steps_html = ""
    for i, step in enumerate(steps):
        name = escape(step.get("step_name", "Unnamed Step"))
        desc = escape(step.get("description", "No description provided"))
        params_data = step.get("parameters", {})
        params_block = (
            f'<div class="sub-section-title">⚙️ Parameters</div>{dict_to_table(params_data, nested=True)}'
            if params_data
            else ""
        )
        output_info = {
            k: v
            for k, v in {
                "Output Shape": step.get("output_shape", "N/A"),
                "Input Columns": step.get("input_columns", "N/A"),
                "Output Columns": step.get("output_columns", "N/A"),
                "Output Dtypes": step.get("output_dtypes", "N/A"),
                "Category Columns": step.get("unique_categories", "N/A"),
            }.items()
            if v != "N/A"
        }
        output_block = (
            f'<div class="sub-section-title">📊 Output Info</div>{dict_to_table(output_info, nested=True)}'
            if output_info
            else ""
        )
        steps_html += f"""<details class="step-accordion"><summary><span class="step-number">{i+1}</span>{name}<span class="chevron">▶</span></summary><div class="step-body"><p class="step-desc">{desc}</p>{params_block}{output_block}</div></details>"""

    # ── Chart Cards ──
    charts_html = '<div class="charts-grid">'

    # ── One-hot range groups ──
    if onehot_groups:
        charts_html += '<div style="grid-column:1/-1" class="subsection-label">🔢 One-Hot Encoded Range Distributions</div>'
    for group_label, bins in onehot_groups.items():
        safe_id = onehot_safe_ids[group_label]
        total_count = 0
        for col_name, lo, hi in bins:
            vals = col_values.get(col_name, [])
            total_count += sum(1 for v in vals if v == 1 or v == 1.0 or v is True)
        n_bins = len(bins)
        range_lo = bins[0][1]
        range_hi = bins[-1][2]
        range_str = f"{range_lo}–{range_hi}"
        charts_html += f"""
        <div class="chart-card">
            <div class="chart-header">
                <h4>{escape(group_label)} <span class="tag tag-onehot">one-hot · {n_bins} bins</span></h4>
                <div class="chart-toggle">
                    <button data-type="rangebar" onclick="renderOH('{safe_id}','rangebar')">Range Bar</button>
                    <button data-type="cumulative" onclick="renderOH('{safe_id}','cumulative')">Cumulative</button>
                    <button data-type="proportion" onclick="renderOH('{safe_id}','proportion')">Proportion</button>
                </div>
            </div>
            <div class="chart-body"><canvas id="oh-{safe_id}"></canvas></div>
            <div class="chart-stats">
                <span class="chart-stat">bins=<strong>{n_bins}</strong></span>
                <span class="chart-stat">range=<strong>{range_str}</strong></span>
                <span class="chart-stat">active=<strong>{total_count}</strong></span>
            </div>
        </div>"""

    # ── Numeric columns ──
    if numeric_cols:
        charts_html += '<div style="grid-column:1/-1" class="subsection-label">📐 Numeric Column Distributions</div>'
    for col in numeric_cols:
        safe_id = "".join(c if c.isalnum() else "_" for c in col)
        esc_col = escape(col).replace("'", "\\'")
        nums = [v for v in col_values[col] if isinstance(v, (int, float))]
        stat_footer = ""
        if nums:
            s_sorted = sorted(nums)
            n = len(s_sorted)
            mean_v = sum(s_sorted) / n
            med_v = s_sorted[n // 2] if n % 2 else (s_sorted[n // 2 - 1] + s_sorted[n // 2]) / 2
            std_v = (sum((x - mean_v) ** 2 for x in s_sorted) / n) ** 0.5
            stat_footer = (
                f'<div class="chart-stats">'
                f'<span class="chart-stat">n=<strong>{n}</strong></span>'
                f'<span class="chart-stat">μ=<strong>{mean_v:.2f}</strong></span>'
                f'<span class="chart-stat">med=<strong>{med_v:.2f}</strong></span>'
                f'<span class="chart-stat">σ=<strong>{std_v:.2f}</strong></span>'
                f'<span class="chart-stat">min=<strong>{s_sorted[0]:.2f}</strong></span>'
                f'<span class="chart-stat">max=<strong>{s_sorted[-1]:.2f}</strong></span>'
                f"</div>"
            )
        charts_html += f"""
        <div class="chart-card">
            <div class="chart-header">
                <h4>{escape(col)} <span class="tag tag-numeric">numeric</span></h4>
                <div class="chart-toggle">
                    <button data-type="histogram" onclick="renderChart('{esc_col}','histogram')">Histogram</button>
                    <button data-type="boxplot" onclick="renderChart('{esc_col}','boxplot')">Box Plot</button>
                    <button data-type="violin" onclick="renderChart('{esc_col}','violin')">Violin</button>
                </div>
            </div>
            <div class="chart-body"><canvas id="chart-{safe_id}"></canvas></div>
            {stat_footer}
        </div>"""

    # ── Categorical columns ──
    if categorical_cols:
        charts_html += '<div style="grid-column:1/-1" class="subsection-label">🏷️ Categorical Column Distributions</div>'
    for col in categorical_cols:
        safe_id = "".join(c if c.isalnum() else "_" for c in col)
        esc_col = escape(col).replace("'", "\\'")
        n_unique = len(set(str(v) for v in col_values[col] if v is not None))
        charts_html += f"""
        <div class="chart-card">
            <div class="chart-header">
                <h4>{escape(col)} <span class="tag tag-categorical">categorical</span></h4>
                <div class="chart-toggle">
                    <button data-type="bar" onclick="renderChart('{esc_col}','bar')">Bar</button>
                    <button data-type="donut" onclick="renderChart('{esc_col}','donut')">Donut</button>
                    <button data-type="pareto" onclick="renderChart('{esc_col}','pareto')">Pareto</button>
                </div>
            </div>
            <div class="chart-body"><canvas id="chart-{safe_id}"></canvas></div>
            <div class="chart-stats"><span class="chart-stat">unique=<strong>{n_unique}</strong></span><span class="chart-stat">total=<strong>{len(col_values[col])}</strong></span></div>
        </div>"""

    # ── Heatmap ──
    if len(corr_cols) >= 2:
        charts_html += (
            '<div style="grid-column:1/-1" class="subsection-label">🔗 Correlation Analysis</div>'
        )
        charts_html += """
        <div class="chart-card heatmap-wrapper">
            <div class="chart-header">
                <h4>Correlation Matrix</h4>
                <div class="chart-toggle">
                    <button data-type="pearson" onclick="drawHeatmap(document.getElementById('heatmap-canvas'),'pearson')">Pearson</button>
                    <button data-type="spearman" onclick="drawHeatmap(document.getElementById('heatmap-canvas'),'spearman')">Spearman</button>
                    <button data-type="kendall" onclick="drawHeatmap(document.getElementById('heatmap-canvas'),'kendall')">Kendall</button>
                </div>
            </div>
            <div class="chart-body"><canvas id="heatmap-canvas"></canvas></div>
            <div class="heatmap-legend">
                <span>-1</span><canvas id="heatmap-legend-grad" width="120" height="12" class="grad"></canvas><span>+1</span>
                <span style="margin-left:auto">● Strong &gt; 0.7 &nbsp; ◐ Moderate 0.4–0.7 &nbsp; ○ Weak &lt; 0.4</span>
            </div>
        </div>"""
    charts_html += "</div>"

    has_viz = bool(numeric_cols or categorical_cols or onehot_groups)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    viz_tab_btn = (
        '<button role="tab" onclick="openTab(event,\'tab-viz\')">📊 Visualizations</button>'
        if has_viz
        else ""
    )
    viz_tab_panel = (
        f"""
    <div id="tab-viz" class="tab-panel" role="tabpanel">
        <div class="section-title"><span class="icon">📊</span> Column Distributions &amp; Correlations</div>
        {charts_html}
    </div>"""
        if has_viz
        else ""
    )

    legend_js = """
    <script>
    window.addEventListener('DOMContentLoaded',()=>{
        const c=document.getElementById('heatmap-legend-grad');
        if(!c)return;const ctx=c.getContext('2d');
        for(let x=0;x<c.width;x++){ctx.fillStyle=PAL.heatmap(x/c.width*2-1);ctx.fillRect(x,0,1,c.height);}
    });
    </script>
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Data Processing Report</title>
{css}{js}{legend_js}
</head>
<body>
<div class="report-header"><div><h1>📈 Data Processing Report</h1><div class="subtitle">Generated {current_time}</div></div></div>
{stats_html}
<div class="tabs-wrapper">
    <div class="tab-bar" role="tablist">
        <button role="tab" onclick="openTab(event,'tab-initial')">📁 Initial Dataset</button>
        <button role="tab" onclick="openTab(event,'tab-steps')">🔧 Processing Steps</button>
        <button role="tab" onclick="openTab(event,'tab-final')">✅ Final Dataset</button>
        {viz_tab_btn}
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
    {viz_tab_panel}
</div>
<div class="report-footer">Pipeline Report &middot; {current_time}</div>
</body></html>"""

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
