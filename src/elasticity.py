"""
esim-pricing-engine / src/elasticity.py
========================================
Price elasticity estimation via log-log OLS, with utilities for
the CR–margin Pareto frontier and optimal price computation.

Model specification (per destination cluster)
---------------------------------------------
log(CR_it) = α_i + β·log(price_it / ref_price_i)
           + γ·log(comp_price_it / price_it)
           + δ·log(seasonality_it)
           + ε_it

Where:
  α_i  = destination fixed effect (within-cluster)
  β    = own-price elasticity  (expected: negative)
  γ    = cross-price elasticity (expected: positive)
  δ    = seasonality coefficient (expected: positive)
  ε_it = i.i.d. noise

We estimate one model per cluster (allowing β to vary by segment)
and one pooled model for comparison.

References
----------
Tellis (1988) "The Price Elasticity of Selective Demand" — meta-analysis
showing median elasticity of -1.76 across consumer goods.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log-transformed features required by the elasticity model.
    Safe log: clips values to avoid log(0).
    """
    df = df.copy()
    eps = 1e-9

    df["log_cr"]           = np.log(df["conversion_rate"].clip(lower=eps))
    df["log_price_ratio"]  = np.log((df["price_usd"] / df["ref_price_usd"]).clip(lower=eps))
    df["log_comp_ratio"]   = np.log((df["comp_price_usd"] / df["price_usd"]).clip(lower=eps))
    df["log_seasonality"]  = np.log(df["seasonality_factor"].clip(lower=eps))
    df["log_price"]        = np.log(df["price_usd"].clip(lower=eps))

    # Month and day-of-week dummies for robustness checks
    df["month"]            = pd.to_datetime(df["date"]).dt.month
    df["dow"]              = pd.to_datetime(df["date"]).dt.dayofweek

    return df


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

FORMULA = "log_cr ~ log_price_ratio + log_comp_ratio + log_seasonality + C(destination_id)"


def fit_cluster_model(
    cluster_df: pd.DataFrame,
    formula: str = FORMULA,
    cluster_name: str = "",
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit log-log OLS for a single destination cluster.
    Destination fixed effects are absorbed via C(destination_id).
    """
    result = smf.ols(formula, data=cluster_df).fit(
        cov_type="HC3"   # heteroskedasticity-robust standard errors
    )
    return result


def fit_all_clusters(df: pd.DataFrame) -> dict:
    """
    Fit one model per cluster + one pooled model.

    Returns
    -------
    dict mapping cluster name → fitted OLS result
    (plus key "POOLED" for the cross-cluster model)
    """
    df = add_log_features(df)
    models = {}

    for cluster, grp in df.groupby("cluster"):
        if grp["destination_id"].nunique() < 2:
            continue
        models[cluster] = fit_cluster_model(grp, cluster_name=cluster)

    # Pooled model — add cluster FE on top of destination FE
    pooled_formula = FORMULA + " + C(cluster)"
    models["POOLED"] = smf.ols(pooled_formula, data=df).fit(cov_type="HC3")

    return models


# ---------------------------------------------------------------------------
# Elasticity extraction
# ---------------------------------------------------------------------------

def extract_elasticities(models: dict) -> pd.DataFrame:
    """
    Extract own-price elasticity (β), cross-price elasticity (γ),
    and seasonality coefficient (δ) with 95% CIs from fitted models.
    """
    rows = []
    for name, res in models.items():
        params = res.params
        ci     = res.conf_int(alpha=0.05)

        def _get(coef):
            if coef not in params.index:
                return np.nan, np.nan, np.nan
            return params[coef], ci.loc[coef, 0], ci.loc[coef, 1]

        beta,  beta_lo,  beta_hi  = _get("log_price_ratio")
        gamma, gamma_lo, gamma_hi = _get("log_comp_ratio")
        delta, delta_lo, delta_hi = _get("log_seasonality")

        rows.append({
            "cluster":        name,
            "n_obs":          int(res.nobs),
            "r2":             res.rsquared,
            "r2_adj":         res.rsquared_adj,
            # Own-price elasticity
            "elasticity":     beta,
            "elast_ci_lo":    beta_lo,
            "elast_ci_hi":    beta_hi,
            # Cross-price elasticity
            "cross_elast":    gamma,
            "cross_ci_lo":    gamma_lo,
            "cross_ci_hi":    gamma_hi,
            # Seasonality
            "season_coef":    delta,
            "season_ci_lo":   delta_lo,
            "season_ci_hi":   delta_hi,
        })

    return pd.DataFrame(rows).set_index("cluster")


# ---------------------------------------------------------------------------
# Pareto frontier: CR vs Margin
# ---------------------------------------------------------------------------

def compute_pareto_frontier(
    base_price: float,
    ref_price: float,
    elasticity: float,
    base_cr: float,
    cost_per_unit: float,
    comp_price: float,
    cross_elast: float = 0.40,
    price_range: tuple = (0.50, 2.0),
    n_points: int = 200,
) -> pd.DataFrame:
    """
    Simulate the CR–margin trade-off curve for a range of prices.

    Parameters
    ----------
    base_price    : current price
    ref_price     : reference price (model intercept anchor)
    elasticity    : own-price elasticity β (negative)
    base_cr       : observed CR at base_price
    cost_per_unit : variable cost per transaction (USD)
    comp_price    : current competitor price
    cross_elast   : cross-price elasticity γ (positive)
    price_range   : (min_multiplier, max_multiplier) relative to ref_price
    n_points      : number of price points to simulate

    Returns
    -------
    DataFrame with columns: price, cr, margin_per_session, revenue_index
    """
    prices = np.linspace(
        ref_price * price_range[0],
        ref_price * price_range[1],
        n_points
    )

    records = []
    for p in prices:
        # Log-log CR prediction
        log_cr = (
            np.log(base_cr)
            + elasticity    * np.log(p / base_price)
            + cross_elast   * np.log(comp_price / p)
        )
        cr     = np.exp(log_cr)
        cr     = np.clip(cr, 0.001, 0.50)

        margin_per_unit    = p - cost_per_unit
        margin_per_session = cr * margin_per_unit   # expected margin per visitor
        revenue_per_session = cr * p

        records.append({
            "price":                p,
            "cr":                   cr,
            "margin_per_unit":      margin_per_unit,
            "margin_per_session":   margin_per_session,
            "revenue_per_session":  revenue_per_session,
            "price_to_ref":         p / ref_price,
        })

    return pd.DataFrame(records)


def find_optimal_price(
    frontier_df: pd.DataFrame,
    objective: str = "margin_per_session",
) -> pd.Series:
    """
    Return the row of frontier_df that maximises the given objective.

    objective options:
        'margin_per_session'   — maximise expected margin per visitor
        'revenue_per_session'  — maximise expected revenue per visitor
        'cr'                   — maximise conversion rate
    """
    assert objective in frontier_df.columns, f"Unknown objective: {objective}"
    return frontier_df.loc[frontier_df[objective].idxmax()]


# ---------------------------------------------------------------------------
# Destination-level summary
# ---------------------------------------------------------------------------

def destination_elasticity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate a simple per-destination elasticity (no FE, just OLS on log-log)
    for visualisation purposes. Quick & dirty — not for inference.
    """
    df = add_log_features(df)
    rows = []

    for dest_id, grp in df.groupby("destination_id"):
        if len(grp) < 30:
            continue
        try:
            res = smf.ols(
                "log_cr ~ log_price_ratio + log_comp_ratio + log_seasonality",
                data=grp
            ).fit()
            rows.append({
                "destination_id": dest_id,
                "cluster":        grp["cluster"].iloc[0],
                "elasticity_est": res.params.get("log_price_ratio", np.nan),
                "elasticity_true": grp["elasticity_true"].iloc[0],
                "r2":             res.rsquared,
                "n_obs":          len(grp),
                "avg_price":      grp["price_usd"].mean(),
                "avg_cr":         grp["conversion_rate"].mean(),
                "avg_margin":     grp["margin_usd"].mean(),
            })
        except Exception:
            continue

    return pd.DataFrame(rows)
