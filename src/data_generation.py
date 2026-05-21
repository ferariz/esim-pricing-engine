"""
esim-pricing-engine / src/data_generation.py
=============================================
Synthetic eSIM transaction data generator.

Design principles
-----------------
- Log-log demand: log(CR) = α + β·log(price/ref_price) + controls
  where β is the price elasticity (typically -1.5 to -3.5 for digital goods)
- Destinations are clustered by traveller profile and price sensitivity
- Seasonality follows real holiday/summer patterns (school calendar + peak travel)
- Competitor price effects: CR rises when you undercut competitors
- Controlled randomness: set SEED for full reproducibility

Usage
-----
    from src.data_generation import generate_dataset
    df = generate_dataset(seed=42)
    df.to_parquet("data/raw/transactions.parquet", index=False)
"""

import numpy as np
import pandas as pd
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
N_DESTINATIONS = 50
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"

# Destination clusters: (cluster_name, n_destinations, base_daily_demand,
#                        ref_price_usd, elasticity_mean, elasticity_std)
DESTINATION_CLUSTERS = [
    # Mature European leisure — elastic, mid-price
    ("europe_leisure",    14, 120, 12.0, -2.2, 0.30),
    # Southeast Asia — very elastic, budget traveller
    ("asia_budget",       10,  90,  8.5, -3.0, 0.35),
    # Americas (US/Canada outbound) — less elastic, premium product
    ("americas_premium",   8, 150, 18.0, -1.6, 0.25),
    # Middle East & Africa — high growth, variable
    ("mea_emerging",       8,  60, 10.0, -2.5, 0.45),
    # Long-haul exotic — niche, low volume, low elasticity
    ("longhaul_exotic",   10,  35, 15.0, -1.8, 0.40),
]

# School/public holiday peaks in Northern Hemisphere (month, day_of_week_agnostic)
# Weight multipliers applied to base demand
HOLIDAY_WEIGHTS = {
    # month: weight
    1:  0.80,   # January: post-Christmas dip
    2:  0.90,   # February: shoulder
    3:  1.05,   # March: spring break starts
    4:  1.15,   # April: Easter, spring break
    5:  1.10,   # May: long weekends
    6:  1.30,   # June: summer begins
    7:  1.55,   # July: peak summer
    8:  1.50,   # August: peak summer
    9:  0.95,   # September: back to school dip
    10: 1.05,   # October: autumn break
    11: 0.85,   # November: quiet
    12: 1.25,   # December: Christmas travel
}

# Day-of-week multipliers (0=Monday … 6=Sunday)
DOW_WEIGHTS = {0: 0.90, 1: 0.85, 2: 0.88, 3: 0.92, 4: 1.10, 5: 1.30, 6: 1.25}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_destinations(rng: np.random.Generator) -> pd.DataFrame:
    """Return a DataFrame with one row per destination and its latent parameters."""
    rows = []
    dest_id = 1
    for cluster, n, base_demand, ref_price, elas_mean, elas_std in DESTINATION_CLUSTERS:
        for i in range(n):
            elasticity = rng.normal(elas_mean, elas_std)
            # Clip to sensible range
            elasticity = np.clip(elasticity, -5.0, -0.5)
            rows.append({
                "destination_id":   f"DEST_{dest_id:03d}",
                "destination_name": f"{cluster}_{i+1:02d}",
                "cluster":          cluster,
                "base_daily_demand": int(rng.normal(base_demand, base_demand * 0.15)),
                "ref_price_usd":    ref_price * rng.uniform(0.85, 1.15),
                "elasticity":       elasticity,
                # Competitor price: typically ±15% of our ref price
                "comp_ref_price":   ref_price * rng.uniform(0.85, 1.15),
                # Intercept noise captures brand/product quality idiosyncrasies
                "alpha_noise":      rng.normal(0, 0.10),
            })
            dest_id += 1
    return pd.DataFrame(rows)


def _compute_seasonality(date: pd.Timestamp) -> float:
    """Combined seasonal multiplier for a given date."""
    month_w = HOLIDAY_WEIGHTS[date.month]
    dow_w   = DOW_WEIGHTS[date.dayofweek]
    return month_w * dow_w


def _compute_cr(
    price: float,
    ref_price: float,
    elasticity: float,
    comp_price: float,
    alpha_noise: float,
    rng: np.random.Generator,
) -> float:
    """
    Log-log conversion rate model.

    CR = exp(α + β·log(price/ref_price) + γ·log(comp_price/price) + ε)

    β (own-price elasticity): negative — higher price → lower CR
    γ (cross-price effect):   positive — competitor is more expensive → our CR rises
    """
    BASE_CR_LOG   = np.log(0.045)   # ~4.5% base conversion at ref price
    GAMMA         = 0.40            # cross-price sensitivity (homogeneous across dests)
    NOISE_SD      = 0.08            # daily noise

    log_cr = (
        BASE_CR_LOG
        + alpha_noise
        + elasticity * np.log(price / ref_price)
        + GAMMA      * np.log(comp_price / price)
        + rng.normal(0, NOISE_SD)
    )
    cr = np.exp(log_cr)
    # Hard-clip to [0.2%, 35%] — physically plausible range
    return float(np.clip(cr, 0.002, 0.35))


def _generate_prices(
    dest: pd.Series,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate our price and competitor price over time.

    Our pricing: ref price ± periodic promotions + random walk drift
    Competitor:  independent random walk around their ref price
    """
    n = len(dates)
    ref = dest["ref_price_usd"]
    comp_ref = dest["comp_ref_price"]

    # Our price: base + small random walk + promotional dips
    our_prices = np.full(n, ref)
    walk = np.cumsum(rng.normal(0, ref * 0.005, n))          # drift
    our_prices = our_prices + walk
    # Clamp to [60%, 160%] of ref
    our_prices = np.clip(our_prices, ref * 0.60, ref * 1.60)

    # Promotions: ~4 per year, randomly placed, -10% to -20%
    n_promos = rng.integers(2, 6)
    promo_starts = rng.integers(0, n - 14, size=n_promos)
    for ps in promo_starts:
        duration = rng.integers(5, 14)
        discount = rng.uniform(0.10, 0.20)
        end = min(ps + duration, n)
        our_prices[ps:end] *= (1 - discount)

    # Competitor price: independent walk around their ref
    comp_walk = np.cumsum(rng.normal(0, comp_ref * 0.004, n))
    comp_prices = np.clip(comp_ref + comp_walk, comp_ref * 0.65, comp_ref * 1.55)

    return our_prices, comp_prices


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_dataset(seed: int = SEED, verbose: bool = True) -> pd.DataFrame:
    """
    Generate the full synthetic eSIM transactions dataset.

    Returns
    -------
    pd.DataFrame with columns:
        date, destination_id, destination_name, cluster,
        price_usd, comp_price_usd, ref_price_usd,
        base_demand, seasonality_factor, conversion_rate,
        sessions, transactions, revenue_usd, margin_usd, elasticity
    """
    rng = np.random.default_rng(seed)

    destinations = _build_destinations(rng)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    if verbose:
        print(f"Generating {len(destinations)} destinations × {len(dates)} days "
              f"= {len(destinations) * len(dates):,} rows …")

    records = []

    for _, dest in destinations.iterrows():
        our_prices, comp_prices = _generate_prices(dest, dates, rng)

        for i, date in enumerate(dates):
            seasonality  = _compute_seasonality(date)
            price        = our_prices[i]
            comp_price   = comp_prices[i]

            cr = _compute_cr(
                price        = price,
                ref_price    = dest["ref_price_usd"],
                elasticity   = dest["elasticity"],
                comp_price   = comp_price,
                alpha_noise  = dest["alpha_noise"],
                rng          = rng,
            )

            # Sessions: base demand × seasonality + Poisson noise
            base_sessions = dest["base_daily_demand"] * seasonality
            sessions      = int(rng.poisson(max(base_sessions, 1)))
            transactions  = int(round(sessions * cr))

            # Margin: assume cost is 35% of ref price (wholesale eSIM cost)
            cost_per_unit = dest["ref_price_usd"] * 0.35
            revenue       = transactions * price
            margin        = transactions * (price - cost_per_unit)

            records.append({
                "date":               date,
                "destination_id":     dest["destination_id"],
                "destination_name":   dest["destination_name"],
                "cluster":            dest["cluster"],
                "price_usd":          round(price, 2),
                "comp_price_usd":     round(comp_price, 2),
                "ref_price_usd":      round(dest["ref_price_usd"], 2),
                "elasticity_true":    round(dest["elasticity"], 4),   # latent ground truth
                "base_demand":        dest["base_daily_demand"],
                "seasonality_factor": round(seasonality, 4),
                "conversion_rate":    round(cr, 6),
                "sessions":           sessions,
                "transactions":       transactions,
                "revenue_usd":        round(revenue, 2),
                "margin_usd":         round(margin, 2),
            })

    df = pd.DataFrame(records)

    if verbose:
        print(f"Done. Shape: {df.shape}")
        print(f"Total transactions: {df['transactions'].sum():,}")
        print(f"Total revenue:      ${df['revenue_usd'].sum():,.0f}")
        print(f"Avg CR:             {df['conversion_rate'].mean():.2%}")

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib
    out = pathlib.Path("data/raw/transactions.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(seed=SEED, verbose=True)
    df.to_parquet(out, index=False)
    print(f"Saved → {out}")

# Note: parquet requires pyarrow. If unavailable, call generate_dataset()
# directly and save as CSV:  df.to_csv("data/raw/transactions.csv", index=False)
