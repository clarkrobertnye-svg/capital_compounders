#!/usr/bin/env python3
"""
Capital Compounders — Intrinsic Value 5yr CAGR Engine
=====================================================

Synthesized best-of-seven AI model approach for per-share IV CAGR.
Designed for automated screening across 400+ companies.

Architecture:
  - Moat Score: 4-factor weighted (VCR, Franchise Power, ROIC stability, SBC)
  - ROIIC Soft Cap: tanh saturator (asset-light friendly)
  - Fade: Exponential decay toward WACC + moat spread
  - Terminal PE: Gordon-style justified PE with quality factor
  - IRR: Bisection solver on projected cash flows
  - Additive decomposition for subscriber transparency

Integration:
  - Reads capital_compounders_universe.json
  - Uses locked WACC: rf + 3.5% + 1% × min(2, max(0, ND/EBITDA - 0.5))
  - Handles asset-light EPS CAGR fallback (V, MA, MSFT, NVDA)
  - Outputs new columns: iv_cagr, iv_g_org, iv_buyback, iv_div, iv_rerate,
    iv_pe_terminal, iv_moat, iv_quality

Usage:
  python iv_cagr_capital_compounders.py [universe.json] [--output results.json]

Author: Capital Compounders / Claude synthesis
Date: 2026-02-23
"""

import json
import math
import sys
import os
from typing import Dict, Optional, Tuple, List
from datetime import datetime


# ═══════════════════════════════════════════════
# CONFIGURATION — TUNE THESE TO YOUR UNIVERSE
# ═══════════════════════════════════════════════

CONFIG = {
    # Locked WACC parameters (from your system)
    "rf": 0.041,                 # Risk-free rate
    "wacc_base_spread": 0.035,   # Base equity spread
    "wacc_leverage_coeff": 0.01, # Leverage penalty per unit

    # Horizon
    "T": 5,                      # Projection horizon (years)

    # Growth credibility
    "g_progressive_break": 0.12, # Full credit up to 12%
    "g_progressive_rate": 0.50,  # 50% credit above break
    "g_max": 0.185,              # Hard ceiling on organic growth

    # Terminal PE
    "pe_floor": 8.0,
    "pe_ceiling": 40.0,

    # Moat score weights
    "w_vcr": 0.35,
    "w_fp": 0.35,
    "w_stability": 0.20,
    "w_sbc": 0.10,

    # Fade parameters
    "k_roiic_min": 0.25,        # Fade speed: elite moat
    "k_roiic_max": 0.65,        # Fade speed: no moat
    "k_rr": 0.20,               # Reinvestment rate fade (slower)
    "moat_spread_max": 0.08,    # Max terminal excess return (elite moat)

    # Asset-light EPS CAGR fallback threshold
    "eps_fallback_rr_threshold": 0.40,  # Use EPS fallback if RR < 10%

    # Financial company detection
    "financial_sectors": ["Financial Services", "Banks", "Insurance"],
    "financial_sic_prefix": ["60", "61", "62", "63", "64", "65"],

    # Blend weights (IRR vs additive)
    "irr_weight": 0.60,
    "additive_weight": 0.40,
}


# ═══════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def clamp01(x: float) -> float:
    return clamp(x, 0.0, 1.0)

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if abs(b) > 1e-9 else default

def safe_float(val, default=0.0) -> float:
    """Safely convert any value to float."""
    if val is None:
        return default
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except (ValueError, TypeError):
        return default


# ═══════════════════════════════════════════════
# WACC — LOCKED FORMULA
# ═══════════════════════════════════════════════

def calc_wacc(nd_ebitda: float, rf: float = None) -> float:
    """
    Locked WACC: rf + 3.5% + 1% × min(2, max(0, ND/EBITDA - 0.5))
    """
    if rf is None:
        rf = CONFIG["rf"]
    leverage_penalty = CONFIG["wacc_leverage_coeff"] * min(2.0, max(0.0, nd_ebitda - 0.5))
    return rf + CONFIG["wacc_base_spread"] + leverage_penalty


# ═══════════════════════════════════════════════
# MOAT SCORE
# ═══════════════════════════════════════════════

def moat_score(vcr: float, franchise_power: float, roic_std: float, sbc_rev: float) -> float:
    """
    Moat durability score m ∈ [0, 1].
    
    s1 (VCR):       VCR 1.0→0, VCR 2.0→1
    s2 (Franchise):  FP 0.03→0, FP 0.15→1  
    s3 (Stability):  σ_roic 0→1, 0.08→0
    s4 (SBC trust):  SBC/Rev 0→1, 0.05→0
    """
    s1 = clamp01((vcr - 1.0) / 1.0)
    s2 = clamp01((franchise_power - 0.03) / 0.12)
    s3 = 1.0 - clamp01(roic_std / 0.08)
    s4 = 1.0 - clamp01(sbc_rev / 0.05)

    return (CONFIG["w_vcr"] * s1 +
            CONFIG["w_fp"] * s2 +
            CONFIG["w_stability"] * s3 +
            CONFIG["w_sbc"] * s4)


# ═══════════════════════════════════════════════
# ROIIC SOFT CAP (tanh saturator)
# ═══════════════════════════════════════════════

def soft_cap_roiic(roiic0: float, m: float) -> float:
    """Moat-dependent soft cap: 60% (no moat) to 150% (elite moat)."""
    cap = 0.60 + 0.90 * m
    if cap < 1e-6:
        return 0.0
    return cap * math.tanh(roiic0 / cap)


# ═══════════════════════════════════════════════
# FADE MECHANICS
# ═══════════════════════════════════════════════

def fade_path(x0: float, x_inf: float, k: float, t: float) -> float:
    return x_inf + (x0 - x_inf) * math.exp(-k * t)

def fade_speed_roiic(m: float) -> float:
    return CONFIG["k_roiic_min"] + (CONFIG["k_roiic_max"] - CONFIG["k_roiic_min"]) * (1.0 - m)

def terminal_roiic_target(wacc: float, m: float) -> float:
    return wacc + CONFIG["moat_spread_max"] * m

def terminal_reinvestment(m: float) -> float:
    return clamp(0.15 + 0.40 * m, 0.10, 0.55)


# ═══════════════════════════════════════════════
# QUALITY FACTOR & CYCLICALITY
# ═══════════════════════════════════════════════

def cyclicality_penalty(roic_std: float) -> float:
    return clamp01(roic_std / 0.10)

def quality_factor(m: float, fcf_conv: float, nd_ebitda: float, cyc: float) -> float:
    """q ∈ [0.70, 2.00] — nonlinear reward for elite moats."""
    q = 0.70 + 0.80 * m + 0.50 * m * m
    q += 0.10 * (fcf_conv - 1.0)
    q -= 0.08 * max(0.0, nd_ebitda - 2.0)
    q *= (1.0 - 0.15 * cyc)
    return clamp(q, 0.70, 2.00)


# ═══════════════════════════════════════════════
# PROGRESSIVE GROWTH CAP
# ═══════════════════════════════════════════════

def progressive_growth_cap(g: float) -> float:
    """
    Up to 12%: full credit
    12-25%: 50% credit
    Max: 18.5%
    """
    brk = CONFIG["g_progressive_break"]
    rate = CONFIG["g_progressive_rate"]
    mx = CONFIG["g_max"]

    if g <= brk:
        return clamp(g, -0.05, mx)
    excess = min(g - brk, 0.13)
    return clamp(brk + rate * excess, -0.05, mx)


# ═══════════════════════════════════════════════
# IRR SOLVER (bisection, no dependencies)
# ═══════════════════════════════════════════════

def solve_irr(cashflows: list, lo: float = -0.50, hi: float = 0.80,
              tol: float = 1e-6, max_iter: int = 200) -> float:
    def npv(rate):
        total = 0.0
        for t, cf in enumerate(cashflows):
            total += cf / (1.0 + rate) ** t
        return total

    if npv(lo) < 0:
        return lo
    if npv(hi) > 0:
        return hi

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if npv(mid) > 0:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < tol:
            break

    return (lo + hi) / 2.0


# ═══════════════════════════════════════════════
# MAIN ENGINE: OPERATING COMPANIES
# ═══════════════════════════════════════════════

def iv_cagr_operating(
    pe0: float, wacc: float, roiic0: float, rr0: float,
    div_yield: float, net_share_delta: float,
    fcf_conv: float, nd_ebitda: float,
    franchise_power: float, vcr: float,
    roic_std: float, sbc_rev: float = 0.0,
    eps_cagr_fallback: float = None,
    T: int = None,
) -> Tuple[float, Dict]:
    """
    Compute 5yr IV CAGR for operating companies.

    If eps_cagr_fallback is provided AND rr0 < threshold (asset-light),
    uses max(ROIIC×RR derived growth, eps_cagr_fallback) as the growth anchor.
    """
    if T is None:
        T = CONFIG["T"]

    # ── Moat Score ──
    m = moat_score(vcr, franchise_power, roic_std, sbc_rev)

    # ── Required return (moat-dependent equity wedge) ──
    eff_wedge = 0.005 + 0.015 * (1.0 - m)
    r = wacc + eff_wedge

    # ── Soft-cap ROIIC ──
    roiic_eff0 = soft_cap_roiic(roiic0, m)

    # ── Asset-light EPS CAGR fallback ──
    # For companies like V, MA, NVDA where accounting reinvestment understates growth
    use_eps_fallback = False
    if (eps_cagr_fallback is not None and
        eps_cagr_fallback > 0 and
        rr0 < CONFIG["eps_fallback_rr_threshold"]):
        # Use 50% haircut on analyst growth as floor (conservative)
        eps_floor = eps_cagr_fallback * 0.50
        roiic_derived_growth = roiic_eff0 * max(0.0, rr0)
        if eps_floor > roiic_derived_growth:
            # Back-solve: what effective ROIIC produces this growth at current RR?
            if rr0 > 0.01:
                roiic_eff0 = eps_floor / rr0
            else:
                # RR ~0: treat entire eps growth as organic, via high effective ROIIC
                roiic_eff0 = eps_floor / 0.10  # assume 10% notional RR
                rr0 = 0.10
            roiic_eff0 = soft_cap_roiic(roiic_eff0, m)
            use_eps_fallback = True

    # ── Fade targets and speeds ──
    roiic_inf = terminal_roiic_target(wacc, m)
    k_roiic = fade_speed_roiic(m)
    rr_inf = terminal_reinvestment(m)
    k_rr = CONFIG["k_rr"] + 0.05 * (1.0 - m)  # slight moat dependence

    # ── Cyclicality ──
    c = cyclicality_penalty(roic_std)

    # ── Year-by-year growth path ──
    yearly_growth = []
    yearly_roiic = []
    yearly_rr = []

    for t in range(1, T + 1):
        roiic_t = fade_path(roiic_eff0, roiic_inf, k_roiic, t)
        rr_t = fade_path(rr0, rr_inf, k_rr, t)
        g_t = roiic_t * max(0.0, rr_t)
        g_t *= (1.0 - 0.30 * c)
        g_t = progressive_growth_cap(g_t)

        yearly_growth.append(g_t)
        yearly_roiic.append(roiic_t)
        yearly_rr.append(rr_t)

    g_avg = sum(yearly_growth) / T

    # ── Terminal growth and payout ──
    roiic_T = yearly_roiic[-1]
    rr_T = yearly_rr[-1]
    g_T_raw = roiic_T * max(0.0, rr_T) * (1.0 - 0.30 * c)
    g_T_raw = progressive_growth_cap(g_T_raw)

    g_T_cap = 0.04 + 0.03 * m
    g_T_organic = min(g_T_raw, r - 0.02, g_T_cap)

    # Buyback accretion in terminal growth
    buyback_rate = min(0.08, max(0.0, -net_share_delta))
    buyback_terminal = buyback_rate * 0.70
    g_T = min(g_T_organic + buyback_terminal, r - 0.02)

    # Terminal payout derived from organic growth & ROIIC
    if roiic_T > 0.01:
        implied_rr_T = clamp(g_T_organic / roiic_T, 0.0, 0.90)
    else:
        implied_rr_T = 0.0
    p_T = clamp(1.0 - implied_rr_T, 0.10, 1.0)

    # ── Quality factor and terminal PE ──
    q = quality_factor(m, fcf_conv, nd_ebitda, c)
    denom = max(r - g_T, 0.005)
    pe_raw = q * (p_T / denom)
    pe_ceiling_moat = 15.0 + 15.0 * m  # moat ceiling: 15x (no moat) to 30x (elite)
    pe_T = clamp(pe_raw, CONFIG["pe_floor"], pe_ceiling_moat)
    # Limit total multiple expansion to 2.0x
    if pe0 > 0:
        pe_T = min(pe_T, pe0 * 2.0)    # ── Additive decomposition ──
    shy = div_yield
    disy = max(0.0, -rr0) * safe_div(1.0, pe0)
    shy += disy

    g_avg_total = g_avg + buyback_rate
    rerate_raw = (pe_T / max(pe0, 1.0)) ** (1.0 / T) - 1.0
    rerate = min(rerate_raw, 0.15)  # cap upside rerate at +15%/yr; keep full downside
    cagr_additive = g_avg_total + shy + rerate

    # ── IRR solve on projected cash flows ──
    div_payout = clamp(div_yield * pe0, 0.0, 0.80)
    eps = [1.0]
    for t in range(T):
        eps.append(eps[-1] * (1.0 + yearly_growth[t] + buyback_rate))

    cashflows = [-pe0]
    for t in range(1, T + 1):
        dividend = eps[t] * div_payout
        if t < T:
            cashflows.append(dividend)
        else:
            terminal_eps = eps[T] * (1.0 + g_T)
            terminal_price = pe_T * terminal_eps
            cashflows.append(dividend + terminal_price)

    cagr_irr = solve_irr(cashflows)

    # ── Final blend ──
    cagr_final = CONFIG["irr_weight"] * cagr_irr + CONFIG["additive_weight"] * cagr_additive
    cagr_final = clamp(cagr_final, -0.50, 1.50)  # allow up to 150%

    return cagr_final, {
        "iv_cagr": round(cagr_final * 100, 2),
        "iv_cagr_irr": round(cagr_irr * 100, 2),
        "iv_cagr_additive": round(cagr_additive * 100, 2),
        "iv_g_organic": round(g_avg * 100, 2),
        "iv_buyback": round(buyback_rate * 100, 2),
        "iv_div": round(shy * 100, 2),
        "iv_rerate": round(rerate * 100, 2),
        "iv_pe_terminal": round(pe_T, 1),
        "pe0": round(pe0, 2),
        "iv_g_terminal": round(g_T * 100, 2),
        "iv_moat": round(m, 3),
        "iv_quality": round(q, 3),
        "iv_cyclicality": round(c, 3),
        "iv_roiic_eff0": round(roiic_eff0 * 100, 1),
        "iv_roiic_terminal": round(roiic_T * 100, 1),
        "iv_eps_fallback_used": use_eps_fallback,
        "iv_growth_path": [round(g * 100, 1) for g in yearly_growth],
    }


# ═══════════════════════════════════════════════
# FINANCIAL COMPANIES ENGINE
# ═══════════════════════════════════════════════

def iv_cagr_financial(
    pe0: float, roe: float, retention: float, wacc: float,
    net_share_delta: float, div_yield: float,
    roic_std: float = 0.03, sbc_rev: float = 0.0,
    T: int = None,
) -> Tuple[float, Dict]:
    """IV CAGR for banks/insurers using ROE-based framework."""
    if T is None:
        T = CONFIG["T"]

    coe = wacc + 0.01  # smaller wedge for financials

    # Simplified moat (ROE vs COE)
    m = clamp01((roe / max(coe, 0.01) - 1.0) / 1.0) * 0.7
    m += 0.3 * (1.0 - clamp01(roic_std / 0.05))
    m = clamp01(m)

    # Fade ROE
    roe_inf = coe + 0.06 * m
    k = 0.20 + 0.40 * (1.0 - m)
    ret_inf = 0.50
    k_ret = 0.20

    yearly_growth = []
    for t in range(1, T + 1):
        roe_t = fade_path(roe, roe_inf, k, t)
        ret_t = fade_path(retention, ret_inf, k_ret, t)
        g_t = clamp(roe_t * ret_t, 0.0, 0.20)
        yearly_growth.append(g_t)

    g_avg = sum(yearly_growth) / T

    roe_T = fade_path(roe, roe_inf, k, T)
    ret_T = fade_path(retention, ret_inf, k_ret, T)
    g_T = min(roe_T * ret_T, coe - 0.02, 0.05)
    p_T = clamp(1.0 - ret_T, 0.0, 1.0)

    q = clamp(0.85 + 0.25 * m, 0.75, 1.15)
    denom = max(coe - g_T, 0.005)
    ptbv_T = clamp(1.0 + q * (roe_T - coe) / denom, 0.70, 2.50)
    pe_T = clamp(ptbv_T / max(roe_T, 0.01), 6.0, 25.0)
    if pe0 > 0:
        pe_T = min(pe_T, pe0 * 2.0)  # 2x expansion cap (financial path)
    buyback_rate = max(0.0, -net_share_delta)
    shy = div_yield
    rerate_raw = (pe_T / max(pe0, 1.0)) ** (1.0 / T) - 1.0
    rerate = min(rerate_raw, 0.15)  # cap upside rerate at +15%/yr; keep full downside
    cagr = g_avg + buyback_rate + shy + rerate

    return cagr, {
        "iv_cagr": round(cagr * 100, 2),
        "iv_g_organic": round(g_avg * 100, 2),
        "iv_buyback": round(buyback_rate * 100, 2),
        "iv_div": round(shy * 100, 2),
        "iv_rerate": round(rerate * 100, 2),
        "iv_pe_terminal": round(pe_T, 1),
        "pe0": round(pe0, 2),
        "iv_moat": round(m, 3),
        "iv_quality": round(q, 3),
        "iv_mode": "financial",
    }


# ═══════════════════════════════════════════════

def compute_roic_operating(t: dict) -> Tuple[float, Dict]:
    """
    Robust ROIC extraction/cleanup for operating companies.

    Returns:
      roic_operating: float (e.g., 0.25 = 25% ROIC)
      flags: dict with metadata about source/cleaning
    """
    flags = {"source": None, "notes": []}

    # Prefer explicit roic if present
    roic_val = t.get("roic")
    roic = safe_float(roic_val, default=0.0)

    roic_hist = t.get("roic_history") or []
    # Clean history into floats
    cleaned = []
    for x in roic_hist:
        fx = safe_float(x, default=float("nan"))
        if math.isfinite(fx):
            cleaned.append(fx)

    # If roic missing/zero, fall back to last history value
    if roic == 0.0 and cleaned:
        roic = cleaned[-1]
        flags["source"] = "roic_history_last"
    elif roic != 0.0:
        flags["source"] = "roic_field"
    elif cleaned:
        # Even if roic was non-finite, use history
        roic = cleaned[-1]
        flags["source"] = "roic_history_last"
    else:
        flags["source"] = "missing"
        roic = 0.0

    # Heuristic: if ROIC is in percent units (e.g., 25 instead of 0.25), normalize
    if roic > 2.5:  # 250% is plausible but rare; 25 is commonly "25%"
        flags["notes"].append("roic_scaled_from_percent")
        roic = roic / 100.0

    # Clamp to sane range (allows high ROIC but avoids absurd parsing errors)
    roic = clamp(roic, -0.50, 3.00)

    # Store cleaned history back if you want downstream std-dev to use it
    flags["roic_hist_cleaned"] = cleaned

    return roic, flags



# DATA EXTRACTION: universe.json → model inputs
# ═══════════════════════════════════════════════

def is_financial(ticker_data: dict) -> bool:
    """Detect financial companies from sector or SIC code."""
    sector = str(ticker_data.get("sector", "")).strip()
    sic = str(ticker_data.get("sic", "")).strip()

    if sector in CONFIG["financial_sectors"] and ticker_data.get("industry", "") not in ["Financial - Credit Services", "Financial - Data & Stock Exchanges", "Financial Exchanges & Data"]:
        return True
    for prefix in CONFIG["financial_sic_prefix"]:
        if sic.startswith(prefix):
            return True
    return False


def extract_inputs(t: dict) -> dict:
    """
    Map Capital Compounders universe.json fields to model inputs.
    Field names matched to actual universe JSON structure (2026-02-23).
    """
    # ── PE: compute from price / (net_income / shares_outstanding) ──
    # ── PE: TTM as ground truth, forward as fallback only ──
    price = safe_float(t.get("price"))
    ni = safe_float(t.get("net_income"))
    shares = safe_float(t.get("shares_outstanding"))

    # 1. Primary: TTM PE from enrichment sources
    pe_ttm = safe_float(t.get("pe_ttm_enriched") or t.get("pe_ttm_yahoo"))
    pe0 = pe_ttm if pe_ttm > 0 else 0.0

    # 2. Fallback: compute from net income if TTM PE not enriched
    if pe0 <= 0 and price > 0 and ni > 0 and shares > 0:
        eps = ni / shares
        if eps > 0:
            pe0 = price / eps

    # 3. Last resort: forward PE only if TTM is still invalid
    fwd_pe = safe_float(t.get("forward_pe"))
    if pe0 <= 0 and fwd_pe > 1.0:
        pe0 = fwd_pe

    # 4. Conservative guard: if forward PE >> TTM, use higher (earnings declining)
    if pe0 > 0 and fwd_pe > 0 and fwd_pe > pe0 * 1.5:
        pe0 = max(pe0, fwd_pe)

    if pe0 < 0 or pe0 > 200:
        pe0 = 0.0  # will skip
    # ── ROIC: from roic_history (most recent = last element) ──
    roic_hist = t.get("roic_history") or []
    roic = roic_hist[-1] if roic_hist else 0.0
    roic = safe_float(roic)

    # ── ROIIC: compute 5yr incremental from nopat_history and ic_history ──
    nopat_hist = t.get("nopat_history") or []
    ic_hist = t.get("ic_history") or []
    roiic = 0.0
    if len(nopat_hist) >= 2 and len(ic_hist) >= 2:
        delta_nopat = safe_float(nopat_hist[-1]) - safe_float(nopat_hist[0])
        delta_ic = safe_float(ic_hist[-1]) - safe_float(ic_hist[0])
        ic_start = abs(safe_float(ic_hist[0]))
        min_delta = max(1e6, 0.05 * ic_start)  # 5% of starting IC, floor $1M
        if abs(delta_ic) > min_delta:
            roiic = delta_nopat / delta_ic
    if roiic == 0.0:
        roiic = roic  # fallback to current ROIC

    # ── Reinvestment rate ──
    rr = safe_float(t.get("reinvestment_rate_3y_avg")) or safe_float(t.get("reinvestment_rate_organic")) or safe_float(t.get("reinvestment_rate"))

    # ── Gross margin ──
    gm = safe_float(t.get("gross_margin"))

    # ── Share changes: net_share_reduction (positive = buyback) ──
    # Model expects negative = buyback, so flip sign
    nsr = clamp(safe_float(t.get("net_share_reduction")), -0.15, 0.15)
    nsd = -nsr  # net_share_reduction 0.02 (2% buyback) → nsd = -0.02

    # ── Dividend yield: compute from fcf_yield × payout proxy ──
    # Not directly in universe — approximate from last_dividend or set 0
    div_y = safe_float(t.get("dividend_yield"))
    if div_y == 0.0:
        # Rough estimate: if fcf_yield exists and payout is small
        fcf_y = safe_float(t.get("fcf_yield"))
        div_y = fcf_y * 0.15  # conservative: assume 15% of FCF paid as dividends

    # ── FCF conversion: ocf_to_net_income ──
    fcf_conv = safe_float(t.get("ocf_to_net_income"))
    if fcf_conv == 0.0:
        fcf_conv = 1.0  # default neutral

    # ── Leverage: compute ND/EBITDA from total_debt, cash, ebitda ──
    total_debt = safe_float(t.get("total_debt"))
    cash = safe_float(t.get("cash_and_equivalents"))
    ebitda = safe_float(t.get("ebitda"))
    nd_ebitda = 0.0
    if ebitda > 0:
        nd_ebitda = max(0.0, (total_debt - cash) / ebitda)

    # ── SBC / Revenue ──
    sbc = safe_float(t.get("sbc"))
    rev = safe_float(t.get("revenue"))
    sbc_rev = sbc / rev if rev > 0 else 0.0


    # ── ROIC: robust operating ROIC (field or roic_history) ──
    roic_operating, roic_flags = compute_roic_operating(t)
    roic = roic_operating
    roic_hist = roic_flags.get("roic_hist_cleaned") or []
    # ── ROIC volatility: compute std from roic_history ──
    roic_std = 0.0
    if len(roic_hist) >= 3:
        import statistics
        try:
            roic_std = statistics.stdev(roic_hist)
        except:
            roic_std = 0.05  # default moderate

    # ── WACC: already in universe (locked formula) ──
    wacc = safe_float(t.get("wacc"))
    if wacc == 0.0:
        wacc = calc_wacc(nd_ebitda)

    # ── Derived metrics ──
    franchise_power = gm * roic if (gm > 0 and roic > 0) else 0.0
    vcr = safe_float(t.get("vcr"))
    if vcr == 0.0 and wacc > 0:
        vcr = safe_div(roic, wacc)

    # ── EPS CAGR fallback (not in universe yet — will come from analyst-estimates) ──
    eps_cagr = safe_float(t.get("eps_cagr_5y") or t.get("eps_next_5y") or t.get("analyst_growth_5y"))
    if eps_cagr > 1.0:
        eps_cagr = eps_cagr / 100.0

    # ── For financial companies ──
    roe = safe_float(t.get("roe"))
    if roe == 0.0 and ni > 0:
        te = safe_float(t.get("total_equity"))
        if te > 0:
            roe = ni / te
    # Retention: 1 - (dividends / net_income), approximate
    retention = 0.70  # default
    if ni > 0 and div_y > 0 and price > 0 and shares > 0:
        annual_divs = div_y * price * shares
        retention = max(0.0, 1.0 - annual_divs / ni)

    return {
        "pe0": pe0,
        "wacc": wacc,
        "roiic0": roiic,
        "roic": roic,
        "rr0": rr,
        "div_yield": div_y,
        "net_share_delta": nsd,
        "fcf_conv": fcf_conv,
        "nd_ebitda": nd_ebitda,
        "franchise_power": franchise_power,
        "vcr": vcr,
        "roic_std": roic_std,
        "sbc_rev": sbc_rev,
        "eps_cagr_fallback": eps_cagr if eps_cagr > 0 else None,
        "gm": gm,
        "roe": roe,
        "retention": retention,
        "is_financial": is_financial(t),
    }


# ═══════════════════════════════════════════════
# PROCESS SINGLE COMPANY
# ═══════════════════════════════════════════════

def process_ticker(ticker: str, ticker_data: dict) -> Optional[Dict]:
    """
    Compute IV CAGR for a single company.
    Returns dict of new columns to merge, or None if insufficient data.
    """
    inp = extract_inputs(ticker_data)

    # Skip if missing critical data
    if inp["pe0"] < 1.0:
        return {"iv_cagr": None, "iv_skip_reason": "no_pe"}
    if inp["wacc"] < 0.01:
        return {"iv_cagr": None, "iv_skip_reason": "no_wacc"}
    if inp["roiic0"] == 0.0 and inp["roic"] == 0.0:
        return {"iv_cagr": None, "iv_skip_reason": "no_roic"}

    try:
        if inp["is_financial"]:
            _, result = iv_cagr_financial(
                pe0=inp["pe0"],
                roe=inp["roe"] if inp["roe"] > 0 else inp["roic"],
                retention=inp["retention"],
                wacc=inp["wacc"],
                net_share_delta=inp["net_share_delta"],
                div_yield=inp["div_yield"],
                roic_std=inp["roic_std"],
                sbc_rev=inp["sbc_rev"],
            )
        else:
            _, result = iv_cagr_operating(
                pe0=inp["pe0"],
                wacc=inp["wacc"],
                roiic0=inp["roiic0"],
                rr0=inp["rr0"],
                div_yield=inp["div_yield"],
                net_share_delta=inp["net_share_delta"],
                fcf_conv=inp["fcf_conv"],
                nd_ebitda=inp["nd_ebitda"],
                franchise_power=inp["franchise_power"],
                vcr=inp["vcr"],
                roic_std=inp["roic_std"],
                sbc_rev=inp["sbc_rev"],
                eps_cagr_fallback=inp["eps_cagr_fallback"],
            )

        result["iv_skip_reason"] = None
        return result

    except Exception as e:
        return {"iv_cagr": None, "iv_skip_reason": f"error: {str(e)[:50]}"}


# ═══════════════════════════════════════════════
# BATCH PROCESSOR
# ═══════════════════════════════════════════════

def process_universe(universe_path: str, output_path: str = None) -> dict:
    """
    Process entire Capital Compounders universe.

    Reads universe.json, computes IV CAGR for each ticker,
    merges results back into the data structure.
    """
    print(f"Loading universe from: {universe_path}")
    with open(universe_path) as f:
        universe = json.load(f)

    # Handle both formats: {"tickers": [...]} or [...]
    if isinstance(universe, dict) and "tickers" in universe:
        tickers = universe["tickers"]
    elif isinstance(universe, list):
        tickers = universe
    else:
        print("ERROR: Unrecognized universe format")
        return {}

    total = len(tickers)
    processed = 0
    skipped = 0
    financials = 0
    eps_fallback = 0
    results_summary = []

    print(f"Processing {total} tickers...")
    print()

    for t in tickers:
        ticker = t.get("symbol") or t.get("ticker") or "???"
        name = t.get("companyName") or t.get("name") or ""

        result = process_ticker(ticker, t)

        if result is None or result.get("iv_cagr") is None:
            skipped += 1
            reason = result.get("iv_skip_reason", "unknown") if result else "no_data"
            t["iv_cagr"] = None
            t["iv_skip_reason"] = reason
            continue

        # Merge results into ticker data
        for key, val in result.items():
            t[key] = val

        processed += 1
        if result.get("iv_mode") == "financial":
            financials += 1
        if result.get("iv_eps_fallback_used"):
            eps_fallback += 1

        results_summary.append({
            "ticker": ticker,
            "name": name[:25],
            "iv_cagr": result["iv_cagr"],
            "pe_T": result.get("iv_pe_terminal"),
            "moat": result.get("iv_moat"),
        })

    # Sort by IV CAGR descending
    results_summary.sort(key=lambda x: x["iv_cagr"] or -999, reverse=True)

    # Print summary
    print(f"{'='*75}")
    print(f"IV CAGR RESULTS — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*75}")
    print(f"  Processed: {processed}/{total}")
    print(f"  Skipped:   {skipped}")
    print(f"  Financial: {financials}")
    print(f"  EPS fallback used: {eps_fallback}")
    print()

    # Top 20
    print(f"{'Rank':<5} {'Ticker':<8} {'Name':<25} {'IV CAGR':>8} {'PE_T':>6} {'Moat':>5}")
    print(f"{'─'*60}")
    for i, r in enumerate(results_summary[:30], 1):
        pe_str = f"{r['pe_T']:.0f}x" if r['pe_T'] else "n/a"
        moat_str = f"{r['moat']:.2f}" if r['moat'] else "n/a"
        print(f"{i:<5} {r['ticker']:<8} {r['name']:<25} {r['iv_cagr']:>7.1f}% {pe_str:>6} {moat_str:>5}")

    if len(results_summary) > 30:
        print(f"  ... {len(results_summary) - 30} more")

    # Bottom 10
    print(f"\n{'─'*60}")
    print("Bottom 10:")
    for r in results_summary[-10:]:
        pe_str = f"{r['pe_T']:.0f}x" if r['pe_T'] else "n/a"
        print(f"  {r['ticker']:<8} {r['name']:<25} {r['iv_cagr']:>7.1f}% {pe_str:>6}")

    # Save results
    if output_path is None:
        output_path = universe_path.replace(".json", "_iv_cagr.json")

    with open(output_path, "w") as f:
        json.dump(universe, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")

    # Also save a lean summary CSV for dashboard
    csv_path = output_path.replace(".json", "_summary.csv")
    with open(csv_path, "w") as f:
        headers = ["ticker", "name", "iv_cagr", "iv_g_organic", "iv_buyback",
                    "iv_div", "iv_rerate", "iv_pe_terminal", "iv_moat", "iv_quality"]
        f.write(",".join(headers) + "\n")
        for t in tickers:
            if t.get("iv_cagr") is not None:
                ticker = t.get("symbol") or t.get("ticker") or ""
                name = (t.get("companyName") or t.get("name") or "").replace(",", "")
                vals = [ticker, name]
                for h in headers[2:]:
                    v = t.get(h, "")
                    vals.append(str(v) if v is not None else "")
                f.write(",".join(vals) + "\n")

    print(f"✅ Summary CSV saved to: {csv_path}")

    return {
        "processed": processed,
        "skipped": skipped,
        "total": total,
        "output_path": output_path,
    }


# ═══════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Capital Compounders IV CAGR Engine")
        print("="*40)
        print()
        print("Usage:")
        print("  python iv_cagr_capital_compounders.py <universe.json> [--output results.json]")
        print()
        print("Or import as module:")
        print("  from iv_cagr_capital_compounders import iv_cagr_operating, process_ticker")
        print()

        # Run demo with archetypes if no file provided
        print("Running archetype demo...")
        print()
        demo_archetypes()
        return

    universe_path = sys.argv[1]
    output_path = None

    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]

    if not os.path.exists(universe_path):
        print(f"ERROR: File not found: {universe_path}")
        sys.exit(1)

    process_universe(universe_path, output_path)


def demo_archetypes():
    """Quick validation against known archetypes."""

    demos = {
        "VISA": dict(pe0=32, wacc=0.085, roiic0=1.72, rr0=0.65,
                      div_yield=0.008, net_share_delta=-0.02,
                      fcf_conv=1.15, nd_ebitda=0.5,
                      franchise_power=0.68*0.47, vcr=0.47/0.085,
                      roic_std=0.03, sbc_rev=0.02),
        "NVDA*": dict(pe0=24.39, wacc=0.076, roiic0=1.45, rr0=0.24,
                       div_yield=0.0002, net_share_delta=-0.028,
                       fcf_conv=0.95, nd_ebitda=0.08,
                       franchise_power=0.70*0.77, vcr=0.77/0.076,
                       roic_std=0.08, sbc_rev=0.035),
        "AAPL": dict(pe0=34, wacc=0.085, roiic0=-0.15, rr0=-0.05,
                      div_yield=0.005, net_share_delta=-0.035,
                      fcf_conv=1.20, nd_ebitda=1.0,
                      franchise_power=0.47*0.82, vcr=0.82/0.085,
                      roic_std=0.04, sbc_rev=0.02),
        "COST": dict(pe0=55, wacc=0.085, roiic0=0.20, rr0=0.18,
                      div_yield=0.006, net_share_delta=-0.005,
                      fcf_conv=1.10, nd_ebitda=0.3,
                      franchise_power=0.13*0.22, vcr=0.22/0.085,
                      roic_std=0.02, sbc_rev=0.01),
    }

    print(f"{'Ticker':<8} {'IV CAGR':>8} {'g_org':>6} {'BB':>5} {'Div':>5} {'Rerate':>7} {'PE_T':>6} {'Moat':>5}")
    print("─" * 60)

    for name, inputs in demos.items():
        _, d = iv_cagr_operating(**inputs)
        print(f"{name:<8} {d['iv_cagr']:>7.1f}% {d['iv_g_organic']:>5.1f}% "
              f"{d['iv_buyback']:>4.1f}% {d['iv_div']:>4.1f}% "
              f"{d['iv_rerate']:>6.1f}% {d['iv_pe_terminal']:>5.0f}x {d['iv_moat']:>5.2f}")

    # Bank
    _, d_bank = iv_cagr_financial(pe0=10, roe=0.15, retention=0.80, wacc=0.10,
                                   net_share_delta=0.005, div_yield=0.03)
    print(f"{'BANK':<8} {d_bank['iv_cagr']:>7.1f}% {d_bank['iv_g_organic']:>5.1f}% "
          f"{d_bank['iv_buyback']:>4.1f}% {d_bank['iv_div']:>4.1f}% "
          f"{d_bank['iv_rerate']:>6.1f}% {d_bank['iv_pe_terminal']:>5.0f}x {d_bank['iv_moat']:>5.2f}")

    print()
    print("* NVDA uses corrected inputs (fwd PE 24.4x, ROIC 145%, RR 24%)")


if __name__ == "__main__":
    main()
