"""
esim-pricing-engine / src/pricing_engine.py
============================================
Core pricing recommendation logic.

The engine takes as input:
  - Current price
  - Price elasticity (from Hito 2)
  - Forecasted daily sessions (from Hito 3)
  - Cost per unit
  - Competitor price
  - Business constraints (min margin, max price change)

And outputs:
  - Recommended price
  - Expected CR at recommended price
  - Expected daily margin at recommended price
  - Reasoning trace (what drove the recommendation)

Three recommendation modes
--------------------------
1. margin_optimal   — maximise expected margin per session
2. revenue_optimal  — maximise expected revenue per session
3. constrained      — maximise margin subject to min_cr floor
                      (useful when CR is a KPI target)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from src.elasticity import compute_pareto_frontier, find_optimal_price


# ---------------------------------------------------------------------------
# Input / Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PricingInput:
    """All inputs required to generate a pricing recommendation."""
    destination_id:    str
    cluster:           str
    current_price:     float       # USD
    ref_price:         float       # USD — model anchor
    elasticity:        float       # own-price elasticity (negative)
    cross_elasticity:  float       # cross-price elasticity (positive)
    comp_price:        float       # competitor current price (USD)
    cost_per_unit:     float       # variable cost per transaction (USD)
    forecasted_sessions: float     # daily sessions forecast
    current_cr:        float       # current observed CR
    # Constraints
    min_margin_pct:    float = 0.20    # floor: margin must be >= 20% of price
    max_price_change:  float = 0.30    # cap: price can move at most ±30%
    min_cr:            float = 0.010   # floor for constrained mode


@dataclass
class PricingRecommendation:
    """Full output of the pricing engine for one destination."""
    destination_id:       str
    cluster:              str
    # Current state
    current_price:        float
    current_cr:           float
    current_margin_per_session: float
    current_daily_margin: float
    # Recommendation
    recommended_price:    float
    recommended_cr:       float
    recommended_margin_per_session: float
    recommended_daily_margin: float
    # Change metrics
    price_change_pct:     float
    cr_change_pct:        float
    margin_uplift_pct:    float
    daily_margin_uplift:  float
    # Metadata
    mode:                 str
    constraint_active:    bool
    reasoning:            list


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def recommend_price(
    inp: PricingInput,
    mode: str = "margin_optimal",
) -> PricingRecommendation:
    """
    Generate a pricing recommendation for one destination.

    Parameters
    ----------
    inp  : PricingInput dataclass
    mode : 'margin_optimal' | 'revenue_optimal' | 'constrained'

    Returns
    -------
    PricingRecommendation dataclass
    """
    assert mode in ("margin_optimal", "revenue_optimal", "constrained"), \
        f"Unknown mode: {mode}"

    # Compute Pareto frontier
    frontier = compute_pareto_frontier(
        base_price    = inp.current_price,
        ref_price     = inp.ref_price,
        elasticity    = inp.elasticity,
        base_cr       = inp.current_cr,
        cost_per_unit = inp.cost_per_unit,
        comp_price    = inp.comp_price,
        cross_elast   = inp.cross_elasticity,
        price_range   = (
            max(0.40, 1 - inp.max_price_change),
            min(2.50, 1 + inp.max_price_change),
        ),
        n_points      = 500,
    )

    # Apply minimum margin constraint
    min_margin_usd = inp.min_margin_pct * frontier["price"]
    frontier = frontier[frontier["margin_per_unit"] >= min_margin_usd].copy()

    # Apply minimum CR constraint (constrained mode)
    constraint_active = False
    if mode == "constrained":
        cr_constrained = frontier[frontier["cr"] >= inp.min_cr]
        if len(cr_constrained) > 0:
            frontier = cr_constrained
            constraint_active = True

    if len(frontier) == 0:
        # Fallback: return current price with a warning
        frontier = compute_pareto_frontier(
            base_price=inp.current_price, ref_price=inp.ref_price,
            elasticity=inp.elasticity, base_cr=inp.current_cr,
            cost_per_unit=inp.cost_per_unit, comp_price=inp.comp_price,
        )

    # Select optimal point
    objective_map = {
        "margin_optimal":  "margin_per_session",
        "revenue_optimal": "revenue_per_session",
        "constrained":     "margin_per_session",
    }
    opt = find_optimal_price(frontier, objective=objective_map[mode])

    # Current state metrics
    curr_margin_per_session = inp.current_cr * (inp.current_price - inp.cost_per_unit)
    curr_daily_margin       = inp.forecasted_sessions * curr_margin_per_session

    # Recommended state metrics
    rec_margin_per_session = opt["cr"] * (opt["price"] - inp.cost_per_unit)
    rec_daily_margin       = inp.forecasted_sessions * rec_margin_per_session

    # Change metrics
    price_change_pct  = (opt["price"] - inp.current_price) / inp.current_price
    cr_change_pct     = (opt["cr"] - inp.current_cr) / inp.current_cr
    margin_uplift_pct = (rec_margin_per_session - curr_margin_per_session) / \
                        max(abs(curr_margin_per_session), 1e-9)
    daily_uplift      = rec_daily_margin - curr_daily_margin

    # Reasoning trace
    reasoning = _build_reasoning(inp, opt, price_change_pct, cr_change_pct,
                                  margin_uplift_pct, mode, constraint_active)

    return PricingRecommendation(
        destination_id              = inp.destination_id,
        cluster                     = inp.cluster,
        current_price               = round(inp.current_price, 2),
        current_cr                  = round(inp.current_cr, 4),
        current_margin_per_session  = round(curr_margin_per_session, 4),
        current_daily_margin        = round(curr_daily_margin, 2),
        recommended_price           = round(float(opt["price"]), 2),
        recommended_cr              = round(float(opt["cr"]), 4),
        recommended_margin_per_session = round(rec_margin_per_session, 4),
        recommended_daily_margin    = round(rec_daily_margin, 2),
        price_change_pct            = round(price_change_pct, 4),
        cr_change_pct               = round(cr_change_pct, 4),
        margin_uplift_pct           = round(margin_uplift_pct, 4),
        daily_margin_uplift         = round(daily_uplift, 2),
        mode                        = mode,
        constraint_active           = constraint_active,
        reasoning                   = reasoning,
    )


def _build_reasoning(inp, opt, price_change_pct, cr_change_pct,
                     margin_uplift_pct, mode, constraint_active):
    """Generate human-readable reasoning for the recommendation."""
    lines = []

    # Elasticity interpretation
    if inp.elasticity < -2.5:
        sens = "highly price-sensitive"
    elif inp.elasticity < -1.8:
        sens = "moderately price-sensitive"
    else:
        sens = "relatively price-inelastic"

    lines.append(
        f"Cluster '{inp.cluster}' is {sens} "
        f"(elasticity β = {inp.elasticity:.2f})."
    )

    # Price direction
    if price_change_pct < -0.02:
        lines.append(
            f"Recommended price cut of {abs(price_change_pct):.1%}: "
            f"high elasticity means the CR gain ({cr_change_pct:+.1%}) "
            f"more than compensates for the lower margin per unit."
        )
    elif price_change_pct > 0.02:
        lines.append(
            f"Recommended price increase of {price_change_pct:.1%}: "
            f"low elasticity means the margin gain outweighs "
            f"the CR cost ({cr_change_pct:+.1%})."
        )
    else:
        lines.append("Current price is near the margin-optimal point. No significant change recommended.")

    # Competitive context
    price_vs_comp = (inp.current_price - inp.comp_price) / inp.comp_price
    if price_vs_comp > 0.10:
        lines.append(
            f"Currently {price_vs_comp:.0%} above competitor price — "
            f"competitive pressure is suppressing CR."
        )
    elif price_vs_comp < -0.10:
        lines.append(
            f"Currently {abs(price_vs_comp):.0%} below competitor price — "
            f"competitive position is strong."
        )

    # Margin uplift
    lines.append(
        f"Expected margin uplift: {margin_uplift_pct:+.1%} per session, "
        f"${inp.forecasted_sessions * (opt['cr'] * (opt['price'] - inp.cost_per_unit) - inp.current_cr * (inp.current_price - inp.cost_per_unit)):.0f}/day "
        f"at {inp.forecasted_sessions:.0f} forecasted sessions."
    )

    # Constraint note
    if constraint_active:
        lines.append(
            f"Min CR floor of {inp.min_cr:.1%} is active — "
            f"unconstrained optimum would price lower."
        )

    return lines


# ---------------------------------------------------------------------------
# Batch recommender — all destinations
# ---------------------------------------------------------------------------

def batch_recommend(
    df: pd.DataFrame,
    elasticity_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    mode: str = "margin_optimal",
    cost_fraction: float = 0.35,
) -> pd.DataFrame:
    """
    Run the pricing engine for all destinations in the dataset.

    Parameters
    ----------
    df            : raw transactions DataFrame
    elasticity_df : output of extract_elasticities(), indexed by cluster
    forecast_df   : DataFrame with [destination_id, sessions_forecast]
    mode          : pricing objective

    Returns
    -------
    DataFrame with one row per destination and recommendation columns
    """
    # Latest snapshot per destination
    latest = (
        df.sort_values("date")
        .groupby("destination_id")
        .last()
        .reset_index()
    )

    # Merge forecast
    if forecast_df is not None:
        latest = latest.merge(
            forecast_df.groupby("destination_id")["sessions_forecast"].mean().reset_index(),
            on="destination_id", how="left"
        )
        latest["sessions_forecast"] = latest["sessions_forecast"].fillna(latest["sessions"])
    else:
        latest["sessions_forecast"] = latest["sessions"]

    rows = []
    for _, row in latest.iterrows():
        cluster = row["cluster"]
        if cluster not in elasticity_df.index:
            continue

        elast_row = elasticity_df.loc[cluster]

        inp = PricingInput(
            destination_id     = row["destination_id"],
            cluster            = cluster,
            current_price      = row["price_usd"],
            ref_price          = row["ref_price_usd"],
            elasticity         = elast_row["elasticity"],
            cross_elasticity   = elast_row.get("cross_elast", 0.40),
            comp_price         = row["comp_price_usd"],
            cost_per_unit      = row["ref_price_usd"] * cost_fraction,
            forecasted_sessions= row["sessions_forecast"],
            current_cr         = row["conversion_rate"],
        )

        rec = recommend_price(inp, mode=mode)
        rows.append({
            "destination_id":        rec.destination_id,
            "cluster":               rec.cluster,
            "current_price":         rec.current_price,
            "recommended_price":     rec.recommended_price,
            "price_change_pct":      rec.price_change_pct,
            "current_cr":            rec.current_cr,
            "recommended_cr":        rec.recommended_cr,
            "cr_change_pct":         rec.cr_change_pct,
            "margin_uplift_pct":     rec.margin_uplift_pct,
            "current_daily_margin":  rec.current_daily_margin,
            "recommended_daily_margin": rec.recommended_daily_margin,
            "daily_margin_uplift":   rec.daily_margin_uplift,
            "mode":                  rec.mode,
        })

    return pd.DataFrame(rows).sort_values("daily_margin_uplift", ascending=False)
