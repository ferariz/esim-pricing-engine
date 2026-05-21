"""
esim-pricing-engine / src/forecasting.py
=========================================
Demand (session volume) forecasting for eSIM pricing engine.

Architecture
------------
Global XGBoost model — one model trained on all destinations simultaneously.
Each destination is identified via target-encoded features and cluster dummies.

This is preferred over per-destination models because:
  1. Low-volume destinations (n < 100 obs) don't have enough data to fit
     a stable individual time-series model.
  2. A global model shares temporal patterns (seasonality, day-of-week) across
     destinations, dramatically reducing overfitting.
  3. It scales to 200+ destinations without retraining N models.

Feature set
-----------
  - Calendar: day_of_year, month, week_of_year, day_of_week, is_weekend
  - Lag features: sessions_lag_7, sessions_lag_14, sessions_lag_28
  - Rolling stats: rolling_mean_7, rolling_mean_28, rolling_std_7
  - Destination: cluster dummies + destination target-encoded mean sessions
  - Price: log_price (pricing affects demand through sessions too)

Train/test split
----------------
  Last 60 days held out as test set — simulates forecasting ahead from a
  real deployment date. All lag features are computed only from training data
  to prevent leakage.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error
from typing import Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_DAYS    = 60     # holdout window
LAG_DAYS     = [7, 14, 28]
ROLLING_WINS = [7, 28]
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based time features."""
    df = df.copy()
    dates = pd.to_datetime(df["date"])

    df["day_of_year"]   = dates.dt.dayofyear
    df["month"]         = dates.dt.month
    df["week_of_year"]  = dates.dt.isocalendar().week.astype(int)
    df["day_of_week"]   = dates.dt.dayofweek
    df["is_weekend"]    = (dates.dt.dayofweek >= 5).astype(int)
    df["quarter"]       = dates.dt.quarter

    # Fourier terms for smooth annual seasonality (avoids dummy trap)
    df["sin_annual"]    = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["cos_annual"]    = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["sin_weekly"]    = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_weekly"]    = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


def build_lag_features(
    df: pd.DataFrame,
    target_col: str = "sessions",
    lag_days: list = LAG_DAYS,
    rolling_wins: list = ROLLING_WINS,
) -> pd.DataFrame:
    """
    Add lag and rolling window features per destination.
    MUST be called after sorting by (destination_id, date).
    """
    df = df.copy().sort_values(["destination_id", "date"])

    grp = df.groupby("destination_id")[target_col]

    for lag in lag_days:
        df[f"lag_{lag}"]  = grp.shift(lag)

    for win in rolling_wins:
        df[f"roll_mean_{win}"] = grp.shift(1).rolling(win).mean().values
        df[f"roll_std_{win}"]  = grp.shift(1).rolling(win).std().values

    return df


def encode_destination_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Encode destination and cluster as numeric features.
    Returns (encoded_df, encoder_dict) for later inference.
    """
    df = df.copy()

    # Cluster label encoding
    le_cluster = LabelEncoder()
    df["cluster_enc"] = le_cluster.fit_transform(df["cluster"])

    # Destination target encoding: mean sessions per destination
    dest_means = df.groupby("destination_id")["sessions"].mean().rename("dest_mean_sessions")
    df = df.merge(dest_means, on="destination_id", how="left")

    encoders = {"cluster": le_cluster, "dest_means": dest_means}
    return df, encoders


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Full feature engineering pipeline. Returns (feature_df, encoders)."""
    df = build_calendar_features(df)
    df = build_lag_features(df)
    df, encoders = encode_destination_features(df)

    # Log price as demand driver
    df["log_price"] = np.log(df["price_usd"].clip(lower=0.01))

    return df, encoders


FEATURE_COLS = [
    # Calendar
    "day_of_year", "month", "week_of_year", "day_of_week",
    "is_weekend", "quarter",
    "sin_annual", "cos_annual", "sin_weekly", "cos_weekly",
    # Lags
    "lag_7", "lag_14", "lag_28",
    # Rolling
    "roll_mean_7", "roll_mean_28", "roll_std_7", "roll_std_28",
    # Destination
    "cluster_enc", "dest_mean_sessions",
    # Price
    "log_price",
]


# ---------------------------------------------------------------------------
# Train / Test Split
# ---------------------------------------------------------------------------

def train_test_split_temporal(
    df: pd.DataFrame,
    test_days: int = TEST_DAYS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal split: last `test_days` calendar days → test.
    All destinations are split at the same date to avoid leakage.
    """
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train  = df[df["date"] <= cutoff].copy()
    test   = df[df["date"] >  cutoff].copy()
    return train, test


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def train_xgb_model(
    train_df: pd.DataFrame,
    feature_cols: list = FEATURE_COLS,
    target_col: str = "sessions",
) -> GradientBoostingRegressor:
    """
    Train a GradientBoostingRegressor on the training set.
    Using sklearn's GBM (no xgboost dependency) — same algorithm,
    swap to xgboost.XGBRegressor in production for speed.
    """
    X = train_df[feature_cols].dropna()
    y = train_df.loc[X.index, target_col]

    model = GradientBoostingRegressor(
        n_estimators    = 400,
        learning_rate   = 0.05,
        max_depth       = 4,
        min_samples_leaf= 10,
        subsample       = 0.8,
        random_state    = RANDOM_STATE,
    )
    model.fit(X, y)
    return model


def predict(
    model: GradientBoostingRegressor,
    df: pd.DataFrame,
    feature_cols: list = FEATURE_COLS,
) -> np.ndarray:
    """Return predictions, clipped to non-negative."""
    X = df[feature_cols].fillna(0)
    preds = model.predict(X)
    return np.clip(preds, 0, None)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_forecasts(
    test_df: pd.DataFrame,
    preds: np.ndarray,
    target_col: str = "sessions",
) -> pd.DataFrame:
    """
    Compute MAPE, RMSE, MAE — overall and per destination cluster.
    """
    df = test_df.copy()
    df["pred"]    = preds
    df["abs_err"] = np.abs(df[target_col] - df["pred"])
    df["pct_err"] = df["abs_err"] / df[target_col].clip(lower=1)

    # Overall
    overall = {
        "cluster":    "OVERALL",
        "mape":       df["pct_err"].mean(),
        "rmse":       np.sqrt((df["abs_err"] ** 2).mean()),
        "mae":        df["abs_err"].mean(),
        "n_obs":      len(df),
    }

    # Per cluster
    rows = [overall]
    for cluster, grp in df.groupby("cluster"):
        rows.append({
            "cluster": cluster,
            "mape":    grp["pct_err"].mean(),
            "rmse":    np.sqrt((grp["abs_err"] ** 2).mean()),
            "mae":     grp["abs_err"].mean(),
            "n_obs":   len(grp),
        })

    return pd.DataFrame(rows).set_index("cluster")


def destination_level_mape(
    test_df: pd.DataFrame,
    preds: np.ndarray,
    target_col: str = "sessions",
) -> pd.DataFrame:
    """MAPE per individual destination — useful for spotting weak spots."""
    df = test_df.copy()
    df["pred"] = preds

    rows = []
    for dest, grp in df.groupby("destination_id"):
        mape = np.abs(grp[target_col] - grp["pred"]).mean() / grp[target_col].clip(lower=1).mean()
        rows.append({
            "destination_id": dest,
            "cluster":        grp["cluster"].iloc[0],
            "mape":           mape,
            "mean_sessions":  grp[target_col].mean(),
            "n_obs":          len(grp),
        })

    return pd.DataFrame(rows).sort_values("mape", ascending=False)


# ---------------------------------------------------------------------------
# Forecast → Pricing bridge
# ---------------------------------------------------------------------------

def forecast_to_pricing_input(
    forecast_df: pd.DataFrame,
    elasticity_df: pd.DataFrame,
    cost_fraction: float = 0.35,
) -> pd.DataFrame:
    """
    Merge demand forecasts with elasticity estimates to produce a
    unified pricing input table.

    Parameters
    ----------
    forecast_df   : DataFrame with columns [date, destination_id, cluster,
                    ref_price_usd, sessions_forecast]
    elasticity_df : Output of extract_elasticities() — indexed by cluster
    cost_fraction : Cost as fraction of ref price

    Returns
    -------
    DataFrame ready to feed into the pricing engine (Hito 5)
    """
    df = forecast_df.merge(
        elasticity_df[["elasticity", "cross_elast"]].reset_index(),
        on="cluster",
        how="left",
    )
    df["cost_per_unit"] = df["ref_price_usd"] * cost_fraction
    df["expected_daily_revenue_at_ref"] = (
        df["sessions_forecast"]
        * df["ref_price_usd"]
        * np.exp(  # predicted CR at ref price
            np.log(0.045) + df["elasticity"] * 0  # log(ref/ref) = 0
        )
    )
    return df
