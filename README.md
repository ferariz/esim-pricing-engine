# eSIM Pricing Engine

> **End-to-end pricing data science for travel eSIM products** — from raw transactions to a deployable recommendation engine.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

<!-- APP SCREENSHOT: Single Destination tab -->
<!-- ![Pricing Engine App](docs/screenshot_app.png) -->

---

## The Problem

A traveller lands in a foreign country. They need mobile data. They search, compare, and buy — or don't — within minutes. For a travel eSIM provider, that moment is the entire business.

Pricing this product is genuinely hard:

- **No recurring revenue** — every transaction stands alone. Mispricing costs the full sale.
- **High price visibility** — travellers compare across providers in seconds. Competitive positioning directly affects conversion.
- **Extreme demand heterogeneity** — a budget backpacker in Southeast Asia and a business traveller on the New York–London corridor have fundamentally different willingness to pay. A single global price rule is wrong for both.
- **Strong seasonality** — July demand can be 2× January demand for the same destination. A price that maximises margin in peak season destroys conversion in the trough.

The central tension in any pricing decision is:

> *Lower price → higher conversion rate → more units sold, but less profit per unit.*
> *Higher price → fatter margin per unit, but fewer buyers.*

Resolving this tension rigorously — not by intuition, but by measurement — is what this project is about.

---

## What I Built

A five-component pricing data science system, built end-to-end in Python:

| Component | What it does | Key output |
|-----------|-------------|------------|
| **Data foundation** | Synthetic eSIM market with realistic demand econometrics | 18,250 rows × 50 destinations × 365 days |
| **Elasticity model** | Log-log OLS per destination cluster | β coefficients with 95% CIs, Pareto frontier |
| **Demand forecast** | Global GBM model with time and lag features | MAPE 16%, Test R² 0.84 |
| **A/B framework** | Power analysis, frequentist z-test, Bayesian Beta-Binomial | Experiment duration calculator, P(B>A) |
| **Pricing engine** | Constraint-aware price optimiser + Streamlit UI | Recommended price with reasoning trace |

---

## The Core Insight: Elasticity Is Not Uniform

The most commercially important finding from this project is also the simplest to state:

**Asia-budget travellers are approximately 2× more price-sensitive than Americas-premium travellers.**

| Cluster | Elasticity β | Interpretation |
|---------|-------------|----------------|
| `asia_budget` | −3.09 | A 10% price rise → 31% CR drop |
| `mea_emerging` | −2.45 | A 10% price rise → 25% CR drop |
| `europe_leisure` | −2.14 | A 10% price rise → 21% CR drop |
| `longhaul_exotic` | −1.74 | A 10% price rise → 17% CR drop |
| `americas_premium` | −1.54 | A 10% price rise → 15% CR drop |

A uniform global price rule systematically overcharges elastic markets (losing conversion) and undercharges inelastic ones (leaving margin on the table). Cluster-aware pricing is not a nice-to-have — it is the baseline requirement for a rational pricing strategy.

---

## The Pareto Frontier

For each destination, there exists a price that maximises expected margin per session — the product of conversion rate and margin per unit. Below that price, you are giving away margin unnecessarily. Above it, you are losing more conversion than the margin gain is worth.

<!-- ![Pareto Frontier](docs/screenshot_pareto.png) -->

The shape of the frontier varies meaningfully by cluster. `americas_premium` has a flat right tail — you can push price quite high before margin per session deteriorates significantly. `asia_budget` collapses fast — the optimal price is lower, and the penalty for over-pricing is severe.

This is the plot I would put in front of a commercial team. It answers the pricing question without requiring them to understand the underlying econometrics.

---

## Demand Forecasting: Why Volume Matters

Elasticity tells you the *rate* — how CR responds to price. But the pricing decision also depends on *volume* — how many sessions you are optimising over.

The same 5% margin improvement at 500 daily sessions (July, peak) is worth 2.5× more in absolute terms than at 200 sessions (January, trough). A pricing engine that ignores forecast volume will systematically misallocate effort.

The global GBM demand model handles all 50 destinations simultaneously, sharing seasonal patterns across the portfolio. Key results:

- **Test R² = 0.84** on a 60-day temporal holdout
- **Overall MAPE = 16.1%** — well within the uncertainty of the elasticity estimates themselves
- Lag features dominate importance: the best predictor of tomorrow's demand is the last 7 and 28 days

---

## Experimentation: Closing the Loop

The pricing engine recommends. The A/B framework validates.

Before rolling out any price change to 100% of traffic, the right process is:

1. **Design**: power analysis to determine required sessions and experiment duration
2. **Run**: split traffic 50/50, collect CR data
3. **Evaluate**: frequentist z-test for the binary decision; Bayesian Beta-Binomial for the probability statement

Two findings worth flagging:

**Traffic is the binding constraint.** At individual destination level (~120 sessions/day), detecting a 10% CR lift requires 924 days. The right unit of experimentation is the cluster (~1,680 sessions/day), where the same effect is detectable in 66 days.

**Peeking destroys statistical validity.** Checking significance daily and stopping at p < 0.05 inflates the false positive rate from the nominal 5% to ~20% — empirically demonstrated across 1,000 simulations.

---

## The Pricing Engine

<!-- ![Streamlit App](docs/screenshot_streamlit.png) -->

The Streamlit app assembles all components into a tool a commercial team can use:

- **Single destination mode**: input any combination of price, elasticity, competitor price, and session forecast — get a recommended price with a Pareto frontier plot and a human-readable reasoning trace
- **Portfolio mode**: one-click batch recommendations across all 50 destinations, ranked by daily margin uplift opportunity
- **Experiment designer**: interactive power analysis calculator — adjust traffic and MDE targets, get experiment duration and design table instantly

The reasoning trace is deliberate. A pricing tool that outputs a number without explanation will not be trusted or adopted. Every recommendation includes: the elasticity regime, competitive context, expected margin uplift, and any active constraints.

---

## Technical Notes

**Why log-log and not linear?** The log-log specification implies constant elasticity — a 1% price change always produces a β% CR change regardless of the price level. This is the standard assumption in demand econometrics and holds well empirically for digital consumer products.

**Why a global GBM over per-destination Prophet?** With 50 destinations and ~365 observations each, per-destination models are prone to overfitting on low-volume series. A global model pools seasonal signal across the portfolio, shares structure, and generalises better.

**Why Bayesian A/B alongside frequentist?** Frequentist tests are the industry default. Bayesian tests give probability statements — *"94% chance the treatment is better"* — which are more actionable for a commercial team making continuous pricing decisions. Both are implemented; the right choice depends on the audience.

**HC3 robust standard errors** throughout the elasticity models — daily CR variance is heteroskedastic on low-traffic destinations, and HC3 gives honest uncertainty estimates without distributional assumptions.

---

## Repository Structure

```
esim-pricing-engine/
├── src/
│   ├── data_generation.py   # Synthetic data engine (log-log DGP)
│   ├── elasticity.py        # Log-log OLS, Pareto frontier
│   ├── forecasting.py       # Global GBM demand forecast
│   ├── ab_testing.py        # Power analysis, z-test, Bayesian evaluator
│   └── pricing_engine.py    # Recommendation engine (3 modes)
├── notebooks/
│   ├── 01_eda.ipynb          # Data foundation & EDA
│   ├── 02_elasticity.ipynb   # Elasticity modelling & Pareto frontier
│   ├── 03_forecasting.ipynb  # Demand forecasting & pricing bridge
│   └── 04_ab_testing.ipynb   # A/B framework & peeking problem
├── app/
│   └── streamlit_app.py      # Interactive pricing UI
├── data/
│   └── raw/                  # Regenerate with: make data
├── requirements.txt
└── Makefile
```

---

## Quick Start

```bash
git clone https://github.com/ferariz/esim-pricing-engine.git
cd esim-pricing-engine
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make data                              # generate synthetic dataset
streamlit run app/streamlit_app.py    # launch pricing engine
```

To run the notebooks:
```bash
jupyter lab notebooks/
```

---

## About

Built as a portfolio project targeting a Senior Data Scientist (Pricing) role in the travel/connectivity vertical.

**Fernando Arizmendi** — [GitHub](https://github.com/ferariz) · [LinkedIn](https://www.linkedin.com/in/fernando-arizmendi/) · [arizmendi.f@gmail.com](mailto:arizmendi.f@gmail.com)
