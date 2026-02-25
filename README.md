# Capital Compounder Investment System v1.0

**Three-tier screening system for identifying 20%+ annual return compounders**

Owner: Rob (clarkrobertnye@gmail.com)  
Based on: Investment Charter v1.0 (January 26, 2026)

---

## Quick Start

### 1. Install Dependencies
```bash
pip install pandas numpy requests
```

### 2. Test with Sample Data
```bash
python run.py --sample
```
This generates a dashboard at `output/capital_compounders_dashboard.html`

### 3. Run with Your Universe (Live FMP API)
```bash
# Using the master universe CSV:
python run.py --live --input capital_compounders_master.csv

# Or specific tickers:
python run.py --live --tickers NVDA MSFT AAPL GOOGL META V MA
```

---

## System Architecture

### Tier 1: Hard Filters (NON-NEGOTIABLE)
Reduces universe from 200+ to ~30-50 candidates.

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Incremental ROIC | ≥ 25% | Each new dollar must earn 25%+ |
| Historical ROIC | ≥ 20% (3Y avg) | Persistent high returns |
| ROIC - WACC Spread | ≥ 15 ppts | Economic profit engine |
| Revenue Growth | ≥ 15% (3Y CAGR) | Cannot compound at 20% with 5% growth |
| FCF Conversion | ≥ 90% (3Y avg) | Cash is king |
| Gross Margin | ≥ 60% (expanding) | Pricing power = moat strength |
| Net Debt/EBITDA | ≤ 2.0× OR Net Cash | Survivability |
| Market Cap | ≥ $10B | Liquidity + sustainability |

### Tier 2: Quality Scoring (100 Points)
Ranks survivors by compounding potential.

| Metric | Weight | Scoring |
|--------|--------|---------|
| Incremental ROIC | 30 pts | >40%=30, 30-40%=20, 25-30%=10 |
| Reinvestment Runway | 20 pts | >$10B=20, $5-10B=15, $1-5B=10 |
| Revenue Growth (3Y) | 20 pts | >20%=20, 15-20%=15, 10-15%=10 |
| FCF Conversion | 15 pts | >100%=15, 95-100%=10, 90-95%=5 |
| Gross Margin Trend | 10 pts | Expanding=10, Stable=5, Declining=0 |
| CapEx Efficiency | 5 pts | <3%=5, 3-5%=3, >5%=0 |

**Tier Labels:**
- EXCEPTIONAL (≥80): Auto-advance to Tier 3
- ELITE (70-79): Advance to Tier 3
- QUALITY (60-69): Watch list only
- REVIEW (<60): Ignore

### Tier 3: DCF Valuation
Calculates intrinsic value and IRR-based entry prices.

**Methodology:**
- 15-year FCFF projection
- Years 1-5: g_val (capped at 20%)
- Years 6-15: Linear fade to 6%
- Terminal: 3% perpetual (GDP ceiling)
- WACC: Risk-free (4%) + Blume-adjusted Beta × ERP (6%)

**Output: Three entry prices**
- Buy@15%: High conviction (25%+ MOS)
- Buy@12%: Watch zone (fair price)
- Buy@10%: Market return

---

## Files Included

| File | Description |
|------|-------------|
| `config.py` | All configurable parameters and API key |
| `fmp_data.py` | FMP API data fetching module |
| `tier1_filter.py` | Hard filter application |
| `tier2_scorer.py` | Quality scoring engine |
| `tier3_valuation.py` | DCF valuation model |
| `dashboard_generator.py` | Interactive HTML dashboard |
| `main.py` | Full pipeline orchestrator |
| `run.py` | Easy run script |
| `sample_data.py` | Sample data for testing |

---

## Configuration

Edit `config.py` to adjust:
- FMP API key
- Filter thresholds
- Scoring weights
- DCF parameters
- Portfolio construction rules

---

## Monthly Operating Cadence

**Week 1: Universe Refresh**
```bash
python run.py --live --input capital_compounders_master.csv --output monthly_run
```

**Week 2-3: Review scored companies and valuations**

**Week 4: Portfolio Review**
- Compare holdings vs Top 20
- Update alerts at Buy@12% prices
- 0-2 trades maximum

---

## Decision Triggers

**BUY (Rare - 3-5 per year):**
- Score ≥ 80 (EXCEPTIONAL)
- IRR ≥ 15%
- MOS ≥ 25%
- No red flags

**HOLD (95% of time):**
- IRR ≥ 10%
- Incremental ROIC ≥ 20%
- Moat stable

**TRIM:**
- IRR < 10%
- Position > 20%

---

## System Health Metrics

✅ Universe size: 30-50 (if >100, filters too loose)  
✅ ELITE+ count: 15-25 (if >40, scoring too lenient)  
✅ BUY signals: 0-2 per month (if >5, standards too low)

---

## Support

For questions or issues, contact: clarkrobertnye@gmail.com

**"The model is not broken when it shows 0 BUYs. The model is broken when it shows BUYs every month."**

Patience is the edge.
