#!/usr/bin/env python3
"""
Capital Compounders Dashboard Generator — V6
=============================================
Reads iv_results.json + capital_compounders_universe.json
Outputs docs/index.html for GitHub Pages deployment.

Columns:
    Ticker | Company | Price | IV CAGR | Organic | Buyback | Div | Rerate |
    PE_T | Moat | Quality | ROIC | VCR | Franchise | ROIIC | ND/EBITDA | EV/EBITDA

Defense-in-Depth (LOCKED):
    TTM PE Priority → Moat Terminal PE → 2x Expansion Cap → 15% Rerate Cap
"""

import csv
import json
import math
import os
from datetime import datetime


def safe_float(v, default=0.0):
    try:
        return float(v) if v is not None else default
    except (ValueError, TypeError):
        return default


def load_data(iv_path="iv_results.json", universe_path="capital_compounders_universe.json"):
    """Load IV results and universe data, merge into company list."""
    with open(iv_path) as f:
        iv_data = json.load(f)
    with open(universe_path) as f:
        uni_data = json.load(f)

    # IV results: dict with 'tickers' key -> nested dict or list
    iv_tickers = iv_data.get("tickers", {})

    # Universe: dict with 'tickers' key -> nested dict or list
    uni_tickers = uni_data.get("tickers", {})

    # Build universe lookup
    if isinstance(uni_tickers, list):
        uni_lookup = {t.get("ticker", ""): t for t in uni_tickers}
    elif isinstance(uni_tickers, dict):
        uni_lookup = {}
        for k, v in uni_tickers.items():
            if isinstance(v, dict):
                uni_lookup[k] = v
            elif isinstance(v, list) and v:
                uni_lookup[k] = v[0] if isinstance(v[0], dict) else {"ticker": k}
            else:
                uni_lookup[k] = {"ticker": k}
    else:
        uni_lookup = {}

    # Build IV results lookup
    if isinstance(iv_tickers, list):
        iv_lookup = {}
        for t in iv_tickers:
            if isinstance(t, dict):
                iv_lookup[t.get("ticker", "")] = t
    elif isinstance(iv_tickers, dict):
        iv_lookup = {}
        for k, v in iv_tickers.items():
            if isinstance(v, dict):
                iv_lookup[k] = v
            elif isinstance(v, list) and v:
                iv_lookup[k] = v[0] if isinstance(v[0], dict) else {}
            else:
                iv_lookup[k] = {}
    else:
        iv_lookup = {}

    # Also try flat CSV as fallback
    csv_path = "iv_results_summary.csv"
    csv_lookup = {}
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                csv_lookup[r.get("ticker", "")] = r

    # Merge
    all_tickers = set(list(iv_lookup.keys()) + list(csv_lookup.keys()))
    companies = []

    for ticker in all_tickers:
        iv = iv_lookup.get(ticker, {})
        u = uni_lookup.get(ticker, {})
        c = csv_lookup.get(ticker, {})

        iv_cagr = safe_float(iv.get("iv_cagr") or c.get("iv_cagr"))
        if iv_cagr == 0 and not iv and not c:
            continue

        # Universe fields
        price = safe_float(u.get("price", 0))
        market_cap = safe_float(u.get("market_cap", 0))
        name = u.get("company_name") or u.get("name", "")
        sector = u.get("sector", "")

        # ROIC
        roic_hist = u.get("roic_history") or []
        roic = safe_float(roic_hist[-1] if roic_hist else u.get("roic", 0))
        roic_pct = roic * 100 if abs(roic) < 1 else roic

        # ROIIC
        roiic_raw = u.get("incremental_roic_5y") or u.get("incremental_roic_3y") or u.get("roiic_5y") or u.get("roiic_3yr")
        if roiic_raw is not None:
            roiic_pct = safe_float(roiic_raw) * 100 if abs(safe_float(roiic_raw)) < 10 else safe_float(roiic_raw)
        else:
            roiic_pct = None

        # Gross margin
        gm = safe_float(u.get("gross_margin", 0))
        gm_pct = gm * 100 if 0 < gm < 1 else gm

        # Franchise Power = GM × ROIC
        franchise = (gm_pct / 100) * (roic_pct / 100) * 100 if gm_pct and roic_pct else 0

        # WACC & VCR
        wacc = safe_float(u.get("wacc", 0))
        wacc_pct = wacc * 100 if 0 < wacc < 1 else wacc
        vcr = roic_pct / wacc_pct if wacc_pct and wacc_pct > 0 and roic_pct else 0

        # Net Debt / EBITDA
        total_debt = safe_float(u.get("total_debt", 0))
        cash = safe_float(u.get("cash_and_equivalents", 0))
        ebitda = safe_float(u.get("ebitda", 0))
        if ebitda > 0:
            net_debt_ebitda = (total_debt - cash) / ebitda
        else:
            net_debt_ebitda = safe_float(u.get("net_debt_to_ebitda") or u.get("net_debt_ebitda"))

        # EV/EBITDA
        if ebitda > 0 and market_cap > 0:
            ev_ebitda = (market_cap + total_debt - cash) / ebitda
        else:
            ev_ebitda = safe_float(u.get("ev_to_ebitda") or u.get("ev_ebitda"))

        # SBC/Rev
        sbc = safe_float(u.get("sbc", 0))
        revenue = safe_float(u.get("revenue", 0))
        sbc_rev = (sbc / revenue * 100) if revenue > 0 and sbc > 0 else safe_float(u.get("sbc_to_revenue", 0)) * 100

        # FCF Conversion
        ocf = safe_float(u.get("operating_cash_flow", 0))
        ni = safe_float(u.get("net_income", 0))
        fcf_conv = (ocf / ni * 100) if ni > 0 and ocf else 0

        companies.append({
            "ticker": ticker,
            "name": name[:28],
            "sector": sector,
            "price": price,
            "market_cap": market_cap,
            "iv_cagr": iv_cagr,
            "iv_g_organic": safe_float(iv.get("iv_g_organic") or c.get("iv_g_organic")),
            "iv_buyback": safe_float(iv.get("iv_buyback") or c.get("iv_buyback")),
            "iv_div": safe_float(iv.get("iv_div") or c.get("iv_div")),
            "iv_rerate": safe_float(iv.get("iv_rerate") or c.get("iv_rerate")),
            "iv_pe_terminal": safe_float(iv.get("iv_pe_terminal") or c.get("iv_pe_terminal")),
            "iv_moat": safe_float(iv.get("iv_moat") or c.get("iv_moat")),
            "iv_quality": safe_float(iv.get("iv_quality") or c.get("iv_quality")),
            "roic": roic_pct,
            "vcr": vcr,
            "franchise": franchise,
            "roiic_pct": roiic_pct,
            "gm": gm_pct,
            "fcf_conv": fcf_conv,
            "sbc_rev": sbc_rev,
            "net_debt_ebitda": net_debt_ebitda,
            "ev_ebitda": ev_ebitda,
        })

    companies.sort(key=lambda x: -x["iv_cagr"])
    return companies


def generate_html(companies):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    total = len(companies)
    positive = sum(1 for c in companies if c["iv_cagr"] > 0)
    buy_zone = sum(1 for c in companies if c["iv_cagr"] >= 20)
    hold_zone = sum(1 for c in companies if 12 <= c["iv_cagr"] < 20)
    rated = [c for c in companies if c["iv_cagr"] != 0]
    avg_iv = sum(c["iv_cagr"] for c in rated) / len(rated) if rated else 0
    avg_roic = sum(c["roic"] for c in companies if c["roic"] > 0) / max(1, sum(1 for c in companies if c["roic"] > 0))

    # Build table rows
    rows_html = ""
    for c in companies:
        t = c["ticker"]

        # Price
        price_s = f"${c['price']:.0f}" if c['price'] > 0 else "—"

        # IV CAGR
        iv = c["iv_cagr"]
        if iv >= 20:
            iv_cls = "irr-high"
        elif iv >= 12:
            iv_cls = "irr-mid"
        elif iv > 0:
            iv_cls = ""
        else:
            iv_cls = "irr-low"
        iv_s = f"{iv:.1f}%"

        # Decomposition
        org_s = f"{c['iv_g_organic']:.1f}%"
        org_cls = "positive" if c['iv_g_organic'] > 0 else "negative" if c['iv_g_organic'] < 0 else ""
        bb_s = f"{c['iv_buyback']:.1f}%"
        bb_cls = "positive" if c['iv_buyback'] > 0 else ""
        div_s = f"{c['iv_div']:.1f}%"
        div_cls = "positive" if c['iv_div'] > 0 else ""
        rr_s = f"{c['iv_rerate']:.1f}%"
        rr_cls = "positive" if c['iv_rerate'] > 0 else "negative" if c['iv_rerate'] < 0 else ""

        # PE Terminal
        pe_t = c["iv_pe_terminal"]
        pe_s = f"{pe_t:.0f}x" if pe_t > 0 else "—"

        # Moat
        moat = c["iv_moat"]
        if moat >= 0.8:
            moat_cls = "positive"
        elif moat >= 0.5:
            moat_cls = ""
        else:
            moat_cls = "warning"
        moat_s = f"{moat:.2f}"

        # Quality
        q = c["iv_quality"]
        q_s = f"{q:.2f}" if q > 0 else "—"

        # ROIC
        roic = c["roic"]
        roic_s = f"{roic:.0f}%" if roic else "—"
        roic_cls = "positive" if roic >= 20 else "" if roic >= 10 else "warning"

        # VCR
        vcr = c["vcr"]
        vcr_s = f"{vcr:.1f}x" if vcr > 0 else "—"
        vcr_cls = "positive" if vcr >= 2 else "" if vcr >= 1 else "negative"

        # Franchise
        fp = c["franchise"]
        fp_s = f"{fp:.0f}%" if fp > 0 else "—"
        fp_cls = "positive" if fp >= 20 else "" if fp >= 10 else "warning"

        # ROIIC
        ri = c["roiic_pct"]
        if ri is not None:
            if abs(ri) > 500:
                ri_s = ">500%"
                ri_cls = "extreme"
            else:
                ri_s = f"{ri:.0f}%"
                ri_cls = "positive" if ri > 0 else "negative"
        else:
            ri_s = "—"
            ri_cls = ""

        # Net Debt/EBITDA
        nde = c["net_debt_ebitda"]
        if nde is not None and nde != 0:
            if nde < 0:
                nde_s = "Net Cash"
                nde_cls = "positive"
            else:
                nde_s = f"{nde:.1f}x"
                nde_cls = "positive" if nde < 1.5 else "warning" if nde < 3 else "negative"
        else:
            nde_s = "—"
            nde_cls = ""

        # EV/EBITDA
        ev = c["ev_ebitda"]
        ev_s = f"{ev:.0f}x" if ev and ev > 0 else "—"

        rows_html += f'''
        <tr class="company-row" data-irr="{iv:.1f}" data-roic="{roic:.0f}" data-moat="{moat:.2f}">
            <td class="ticker-cell"><a href="https://finance.yahoo.com/quote/{t}" target="_blank">{t}</a></td>
            <td class="name-cell">{c['name']}</td>
            <td class="metric price">{price_s}</td>
            <td class="metric irr {iv_cls}">{iv_s}</td>
            <td class="metric {org_cls}">{org_s}</td>
            <td class="metric hide-mobile {bb_cls}">{bb_s}</td>
            <td class="metric hide-mobile {div_cls}">{div_s}</td>
            <td class="metric {rr_cls}">{rr_s}</td>
            <td class="metric pe">{pe_s}</td>
            <td class="metric moat {moat_cls}">{moat_s}</td>
            <td class="metric hide-mobile qual">{q_s}</td>
            <td class="metric roic group-start {roic_cls}">{roic_s}</td>
            <td class="metric hide-mobile vcr {vcr_cls}">{vcr_s}</td>
            <td class="metric hide-mobile franchise {fp_cls}">{fp_s}</td>
            <td class="metric hide-mobile roiic {ri_cls}">{ri_s}</td>
            <td class="metric hide-mobile nde {nde_cls}">{nde_s}</td>
            <td class="metric hide-mobile val">{ev_s}</td>
        </tr>'''

    companies_json = json.dumps([{
        "ticker": c["ticker"], "name": c["name"], "price": c["price"],
        "iv_cagr": c["iv_cagr"], "iv_g_organic": c["iv_g_organic"],
        "iv_buyback": c["iv_buyback"], "iv_div": c["iv_div"],
        "iv_rerate": c["iv_rerate"], "iv_pe_terminal": c["iv_pe_terminal"],
        "iv_moat": c["iv_moat"], "iv_quality": c["iv_quality"],
        "roic": c["roic"], "vcr": c["vcr"], "franchise": c["franchise"],
        "roiic_pct": c["roiic_pct"], "gm": c["gm"],
        "net_debt_ebitda": c["net_debt_ebitda"], "ev_ebitda": c["ev_ebitda"],
        "market_cap": c["market_cap"],
    } for c in companies])

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <title>Capital Compounders</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0a0f0f;
            --bg-secondary: #0f1a1a;
            --bg-card: #132020;
            --bg-card-accent: #1a2a2a;
            --text-primary: #e0f0f0;
            --text-secondary: #80a0a0;
            --text-muted: #507070;
            --border: #2a4040;
            --accent-cyan: #00d4aa;
            --accent-green: #00d4aa;
            --accent-yellow: #d4aa00;
            --accent-blue: #00a4d4;
            --accent-red: #ff6b6b;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.5;
        }}

        .container {{ max-width: 1800px; margin: 0 auto; padding: 20px; }}

        header {{
            text-align: center;
            padding: 25px 0 15px;
        }}

        h1 {{
            font-size: 2.2rem;
            color: var(--accent-cyan);
            font-weight: 700;
            letter-spacing: -0.5px;
        }}

        .tagline {{
            color: var(--text-secondary);
            margin-top: 6px;
            font-size: 0.9rem;
        }}

        .defense-tag {{
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: var(--text-muted);
            background: var(--bg-card);
            border: 1px solid var(--border);
            padding: 3px 10px;
            border-radius: 4px;
            margin-top: 8px;
        }}

        .metrics-row {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }}

        .metric-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px;
            text-align: center;
        }}

        .metric-card.highlight {{
            border-color: var(--accent-cyan);
            background: linear-gradient(180deg, var(--bg-card) 0%, rgba(0,212,170,0.08) 100%);
        }}

        .metric-card .label {{
            font-size: 0.72rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}

        .metric-card .value {{
            font-size: 1.5rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent-cyan);
        }}

        .metric-card .sublabel {{
            font-size: 0.68rem;
            color: var(--text-muted);
            margin-top: 2px;
        }}

        .controls-bar {{
            display: flex;
            gap: 12px;
            margin-bottom: 14px;
            align-items: center;
            flex-wrap: wrap;
        }}

        .search-input {{
            flex: 1;
            min-width: 180px;
            padding: 10px 14px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }}

        .search-input:focus {{ outline: none; border-color: var(--accent-cyan); }}
        .search-input::placeholder {{ color: var(--text-muted); }}

        .filter-group {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .filter-group label {{
            color: var(--text-muted);
            font-size: 0.78rem;
        }}

        .filter-group select {{
            background: var(--bg-card);
            color: var(--text-primary);
            border: 1px solid var(--border);
            padding: 8px 10px;
            border-radius: 6px;
            font-size: 0.78rem;
            cursor: pointer;
        }}

        .filter-group select:hover {{ border-color: var(--accent-cyan); }}

        .row-count {{
            color: var(--text-muted);
            font-size: 0.78rem;
            margin-left: auto;
            font-family: 'JetBrains Mono', monospace;
        }}

        .table-container {{
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            border: 1px solid var(--border);
            border-radius: 10px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8rem;
        }}

        th {{
            background: var(--bg-secondary);
            padding: 10px 8px;
            text-align: right;
            font-weight: 600;
            color: var(--accent-cyan);
            border-bottom: 2px solid var(--border);
            cursor: pointer;
            white-space: nowrap;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            position: sticky;
            top: 0;
            z-index: 10;
        }}

        th:first-child, th:nth-child(2) {{ text-align: left; }}
        th:hover {{ background: var(--bg-card); }}
        th.group-start {{ border-left: 2px solid var(--border); }}
        td.group-start {{ border-left: 2px solid rgba(42,64,64,0.5); }}

        th.sort-asc::after {{ content: ' ▲'; font-size: 0.6rem; color: var(--accent-cyan); }}
        th.sort-desc::after {{ content: ' ▼'; font-size: 0.6rem; color: var(--accent-cyan); }}

        td {{
            padding: 7px 8px;
            border-bottom: 1px solid rgba(42,64,64,0.4);
            background: var(--bg-card);
        }}

        tr:hover td {{ background: var(--bg-card-accent); }}

        .ticker-cell {{ text-align: left; }}
        .ticker-cell a {{
            color: var(--accent-cyan);
            text-decoration: none;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
        }}
        .ticker-cell a:hover {{ text-decoration: underline; }}

        .name-cell {{
            color: var(--text-secondary);
            text-align: left;
            font-size: 0.72rem;
            max-width: 160px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .metric {{
            text-align: right;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
        }}

        .metric.irr {{ font-weight: 700; font-size: 0.85rem; }}
        .metric.irr-high {{ color: var(--accent-green); }}
        .metric.irr-mid {{ color: var(--accent-yellow); }}
        .metric.irr-low {{ color: var(--text-muted); }}

        .metric.moat {{ font-weight: 600; }}
        .metric.pe {{ color: var(--text-secondary); }}
        .metric.qual {{ color: var(--text-secondary); }}
        .metric.price {{ color: var(--text-secondary); }}
        .metric.val {{ color: var(--text-secondary); }}
        .metric.roic {{ font-weight: 600; }}

        .metric.positive {{ color: var(--accent-green); }}
        .metric.warning {{ color: var(--accent-yellow); }}
        .metric.negative {{ color: var(--accent-red); }}
        .metric.extreme {{ color: var(--accent-green); font-weight: 700; }}

        .tooltip-box {{
            display: none;
            position: absolute;
            background: #0d1414;
            color: var(--text-primary);
            padding: 10px 14px;
            border: 1px solid var(--accent-cyan);
            border-radius: 6px;
            font-size: 0.78rem;
            max-width: 300px;
            z-index: 9999;
            transform: translate(-50%, -100%);
            pointer-events: none;
            line-height: 1.4;
        }}

        .glossary {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 22px;
            margin-top: 25px;
        }}

        .glossary h2 {{
            color: var(--accent-cyan);
            font-size: 1.1rem;
            margin-bottom: 16px;
        }}

        .glossary-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }}

        .glossary-item {{
            background: var(--bg-card);
            border-left: 3px solid var(--accent-cyan);
            padding: 10px 12px;
            border-radius: 0 6px 6px 0;
        }}

        .glossary-item .term {{
            display: block;
            font-weight: 600;
            color: var(--accent-cyan);
            font-size: 0.82rem;
            margin-bottom: 3px;
        }}

        .glossary-item .formula {{
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: var(--text-secondary);
            background: var(--bg-primary);
            padding: 2px 6px;
            border-radius: 3px;
            margin-bottom: 5px;
        }}

        .glossary-item .desc {{
            font-size: 0.72rem;
            color: var(--text-muted);
            line-height: 1.4;
            margin: 0;
        }}

        footer {{
            text-align: center;
            padding: 25px;
            color: var(--text-muted);
            font-size: 0.78rem;
            font-family: 'JetBrains Mono', monospace;
        }}

        @media (max-width: 1200px) {{
            .metrics-row {{ grid-template-columns: repeat(3, 1fr); }}
            .glossary-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}

        @media (max-width: 768px) {{
            .container {{ padding: 10px; }}
            h1 {{ font-size: 1.4rem; }}
            .metrics-row {{ grid-template-columns: repeat(2, 1fr); gap: 8px; }}
            .metric-card {{ padding: 10px; }}
            .metric-card .value {{ font-size: 1.2rem; }}
            .controls-bar {{ gap: 8px; }}
            table {{ font-size: 0.72rem; }}
            th, td {{ padding: 6px 5px; }}
            .name-cell {{ max-width: 80px; font-size: 0.65rem; }}
            .glossary-grid {{ grid-template-columns: 1fr; }}
            .hide-mobile {{ display: none; }}
        }}

        @media (max-width: 480px) {{
            .metrics-row {{ grid-template-columns: repeat(2, 1fr); }}
            .metric-card .value {{ font-size: 1rem; }}
        }}
    </style>
</head>
<body>
<div class="container">

    <header>
        <h1>CAPITAL COMPOUNDERS</h1>
        <div class="tagline">IV CAGR Rankings &mdash; {total} Companies &mdash; {now}</div>
        <div class="defense-tag">TTM PE &rarr; Moat PE (15+15m) &rarr; 2x Expansion Cap &rarr; 15% Rerate Cap</div>
    </header>

    <div class="metrics-row">
        <div class="metric-card highlight">
            <div class="label">Universe</div>
            <div class="value">{total}</div>
            <div class="sublabel">{positive} positive IV CAGR</div>
        </div>
        <div class="metric-card highlight">
            <div class="label">IV &ge; 20%</div>
            <div class="value">{buy_zone}</div>
            <div class="sublabel">buy zone</div>
        </div>
        <div class="metric-card">
            <div class="label">IV 12-20%</div>
            <div class="value" style="color:var(--accent-yellow)">{hold_zone}</div>
            <div class="sublabel">hold zone</div>
        </div>
        <div class="metric-card">
            <div class="label">Avg IV CAGR</div>
            <div class="value">{avg_iv:.1f}%</div>
            <div class="sublabel">rated universe</div>
        </div>
        <div class="metric-card">
            <div class="label">Avg ROIC</div>
            <div class="value">{avg_roic:.0f}%</div>
            <div class="sublabel">ROIC &gt; 0 universe</div>
        </div>
        <div class="metric-card">
            <div class="label">Top IV CAGR</div>
            <div class="value">{companies[0]['iv_cagr']:.1f}%</div>
            <div class="sublabel">{companies[0]['ticker']}</div>
        </div>
    </div>

    <div class="controls-bar">
        <input type="text" class="search-input" id="searchBox" placeholder="Search ticker or company..." autocomplete="off">
        <div class="filter-group">
            <label>IV CAGR:</label>
            <select id="filterIrr">
                <option value="all">All</option>
                <option value="20">&ge; 20%</option>
                <option value="12">&ge; 12%</option>
                <option value="0">Positive</option>
            </select>
        </div>
        <div class="filter-group">
            <label>Moat:</label>
            <select id="filterMoat">
                <option value="all">All</option>
                <option value="0.80">&ge; 0.80</option>
                <option value="0.60">&ge; 0.60</option>
            </select>
        </div>
        <div class="filter-group">
            <label>ROIC:</label>
            <select id="filterRoic">
                <option value="all">All</option>
                <option value="30">&ge; 30%</option>
                <option value="20">&ge; 20%</option>
                <option value="15">&ge; 15%</option>
            </select>
        </div>
        <span class="row-count" id="rowCount">{total} companies</span>
    </div>

    <div class="table-container">
        <table>
            <thead><tr>
                <th data-tooltip="Stock ticker — click to open Yahoo Finance">Ticker</th>
                <th data-tooltip="Company name">Company</th>
                <th data-tooltip="Current share price">Price</th>
                <th data-tooltip="Intrinsic Value 5yr CAGR — total expected return">IV CAGR</th>
                <th data-tooltip="Organic growth (ROIIC x Reinvestment Rate)">Organic</th>
                <th class="hide-mobile" data-tooltip="Share buyback contribution">Buyback</th>
                <th class="hide-mobile" data-tooltip="Dividend yield contribution">Div</th>
                <th data-tooltip="PE rerating contribution (capped 15%/yr)">Rerate</th>
                <th data-tooltip="Terminal PE after moat ceiling + 2x expansion cap">PE_T</th>
                <th data-tooltip="Moat score (0-1) — drives PE ceiling: 15+15*moat">Moat</th>
                <th class="hide-mobile" data-tooltip="Quality score — composite metric">Quality</th>
                <th class="group-start" data-tooltip="Return on Invested Capital (TTM)">ROIC</th>
                <th class="hide-mobile" data-tooltip="Value Creation Ratio = ROIC / WACC">VCR</th>
                <th class="hide-mobile" data-tooltip="Franchise Power = GM x ROIC">Franchise</th>
                <th class="hide-mobile" data-tooltip="Return on Incremental Invested Capital (5yr)">ROIIC</th>
                <th class="hide-mobile" data-tooltip="Net Debt / EBITDA (negative = net cash)">ND/EB</th>
                <th class="hide-mobile" data-tooltip="Enterprise Value / EBITDA">EV/EB</th>
            </tr></thead>
            <tbody>{rows_html}
            </tbody>
        </table>
    </div>

    <div class="glossary">
        <h2>Methodology</h2>
        <div class="glossary-grid">
            <div class="glossary-item">
                <span class="term">IV CAGR</span>
                <span class="formula">Organic + Buyback + Div + Rerate</span>
                <p class="desc">5-year intrinsic value per-share CAGR. Moat-based terminal PE with 4 defensive guards.</p>
            </div>
            <div class="glossary-item">
                <span class="term">Moat Score</span>
                <span class="formula">0.0 to 1.0</span>
                <p class="desc">Composite quality metric. Drives terminal PE ceiling: PE_max = 15 + 15 &times; moat.</p>
            </div>
            <div class="glossary-item">
                <span class="term">2x Expansion Cap</span>
                <span class="formula">PE_T &le; PE_0 &times; 2.0</span>
                <p class="desc">Terminal PE cannot exceed 2x current PE. Matches 15%/yr rerating over 5 years.</p>
            </div>
            <div class="glossary-item">
                <span class="term">Franchise Power</span>
                <span class="formula">GM &times; ROIC</span>
                <p class="desc">Pricing power &times; capital efficiency. Higher = wider moat.</p>
            </div>
            <div class="glossary-item">
                <span class="term">VCR</span>
                <span class="formula">ROIC / WACC</span>
                <p class="desc">Value Creation Ratio. Above 2x = elite. Below 1x = value destroyer.</p>
            </div>
            <div class="glossary-item">
                <span class="term">WACC</span>
                <span class="formula">rf + 3.5% + 1% &times; min(2, max(0, L-0.5))</span>
                <p class="desc">rf=4.1%, L=max(0, ND/EBITDA). Leverage-adjusted, replaces beta.</p>
            </div>
        </div>
    </div>

    <footer>
        Capital Compounders &bull; {now} &bull; Defense-in-Depth: TTM PE &rarr; Moat PE &rarr; 2x Expansion &rarr; 15% Rerate Cap
    </footer>

</div>

<script>
    const allCompanies = {companies_json};
    const sortState = {{ col: null, asc: false }};

    const searchBox = document.getElementById('searchBox');
    searchBox.addEventListener('input', applyFilters);
    searchBox.addEventListener('input', () => {{ searchBox.value = searchBox.value.toUpperCase(); }});

    document.getElementById('filterIrr').addEventListener('change', applyFilters);
    document.getElementById('filterMoat').addEventListener('change', applyFilters);
    document.getElementById('filterRoic').addEventListener('change', applyFilters);

    function applyFilters() {{
        const q = searchBox.value.trim().toLowerCase();
        const irrF = document.getElementById('filterIrr').value;
        const moatF = document.getElementById('filterMoat').value;
        const roicF = document.getElementById('filterRoic').value;

        let visible = 0;
        document.querySelectorAll('.company-row').forEach(row => {{
            const ticker = row.querySelector('.ticker-cell a')?.textContent?.toLowerCase() || '';
            const name = row.querySelector('.name-cell')?.textContent?.toLowerCase() || '';
            const irr = parseFloat(row.dataset.irr) || 0;
            const roic = parseFloat(row.dataset.roic) || 0;
            const moat = parseFloat(row.dataset.moat) || 0;

            const matchSearch = !q || ticker.includes(q) || name.includes(q);
            const matchIrr = irrF === 'all' || irr >= parseFloat(irrF);
            const matchMoat = moatF === 'all' || moat >= parseFloat(moatF);
            const matchRoic = roicF === 'all' || roic >= parseFloat(roicF);

            const show = matchSearch && matchIrr && matchMoat && matchRoic;
            row.style.display = show ? '' : 'none';
            if (show) visible++;
        }});
        document.getElementById('rowCount').textContent = visible + ' companies';
    }}

    document.querySelectorAll('th').forEach((th, i) => {{
        th.addEventListener('click', () => {{
            const tbody = document.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));

            if (sortState.col === i) {{
                sortState.asc = !sortState.asc;
            }} else {{
                sortState.col = i;
                sortState.asc = false;
            }}

            document.querySelectorAll('th').forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
            th.classList.add(sortState.asc ? 'sort-asc' : 'sort-desc');

            const textCols = [0, 1];
            const isNumeric = !textCols.includes(i);
            rows.sort((a, b) => {{
                let aVal = a.cells[i]?.textContent?.trim() || '';
                let bVal = b.cells[i]?.textContent?.trim() || '';
                if (aVal === '—' || aVal === 'Net Cash') aVal = isNumeric ? (aVal === 'Net Cash' ? '-999' : '0') : aVal;
                if (bVal === '—' || bVal === 'Net Cash') bVal = isNumeric ? (bVal === 'Net Cash' ? '-999' : '0') : bVal;
                if (aVal.startsWith('>')) aVal = aVal.replace('>', '');
                if (bVal.startsWith('>')) bVal = bVal.replace('>', '');
                if (isNumeric) {{
                    aVal = parseFloat(aVal.replace(/[$%x,+]/g, '')) || 0;
                    bVal = parseFloat(bVal.replace(/[$%x,+]/g, '')) || 0;
                    return sortState.asc ? aVal - bVal : bVal - aVal;
                }}
                return sortState.asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
            }});
            rows.forEach(r => tbody.appendChild(r));
        }});
    }});

    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip-box';
    document.body.appendChild(tooltip);

    document.querySelectorAll('th[data-tooltip]').forEach(th => {{
        th.addEventListener('mouseenter', e => {{
            tooltip.textContent = th.dataset.tooltip;
            tooltip.style.display = 'block';
            const rect = th.getBoundingClientRect();
            tooltip.style.left = (rect.left + rect.width/2) + 'px';
            tooltip.style.top = (rect.top + window.scrollY - 10) + 'px';
        }});
        th.addEventListener('mouseleave', () => tooltip.style.display = 'none');
    }});
</script>
</body>
</html>'''


def main():
    iv_path = "iv_results.json"
    uni_path = "capital_compounders_universe.json"

    if not os.path.exists(iv_path):
        print(f"ERROR: {iv_path} not found")
        return
    if not os.path.exists(uni_path):
        print(f"ERROR: {uni_path} not found")
        return

    companies = load_data(iv_path, uni_path)
    print(f"Loaded {{len(companies)}} companies")

    os.makedirs("docs", exist_ok=True)
    html = generate_html(companies)

    out = "docs/index.html"
    with open(out, "w") as f:
        f.write(html)

    rated = [c for c in companies if c["iv_cagr"] != 0]
    buy = sum(1 for c in companies if c["iv_cagr"] >= 20)
    hold = sum(1 for c in companies if 12 <= c["iv_cagr"] < 20)

    print(f"Dashboard v6 saved to {{out}}")
    print(f"   Universe: {{len(companies)}} | Rated: {{len(rated)}}")
    print(f"   IV>=20%: {{buy}} | 12-20%: {{hold}}")
    print(f"   Open: file://{{os.path.abspath(out)}}")


if __name__ == "__main__":
    main()
