"""
esim-pricing-engine / app/streamlit_app.py
==========================================
Pricing Recommendation Engine — Interactive UI

Run with:
    streamlit run app/streamlit_app.py

Three tabs:
  1. Single Destination — input parameters manually, get a recommendation
  2. Portfolio Overview — run batch recommendations on the full dataset
  3. Experiment Designer — A/B test power analysis calculator
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import streamlit as st

from src.elasticity import (
    add_log_features, fit_all_clusters,
    extract_elasticities, compute_pareto_frontier,
)
from src.pricing_engine import PricingInput, recommend_price, batch_recommend
from src.ab_testing import (
    sample_size_per_variant, experiment_duration_days,
    mde_from_sample_size, design_summary,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title  = "eSIM Pricing Engine",
    page_icon   = "📡",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# Data & model loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    data_path = Path(__file__).parent.parent / "data" / "raw" / "transactions.csv"
    df = pd.read_csv(data_path, parse_dates=["date"])
    return df

@st.cache_data
def load_elasticities(_df):
    df_feat = add_log_features(_df)
    models  = fit_all_clusters(df_feat)
    elast   = extract_elasticities(models)
    return elast

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.image("https://img.icons8.com/fluency/96/sim-card.png", width=60)
st.sidebar.title("eSIM Pricing Engine")
st.sidebar.caption("Portfolio project · ferariz/esim-pricing-engine")
st.sidebar.divider()

tab_labels = ["📍 Single Destination", "🗺️ Portfolio Overview", "🧪 Experiment Designer"]
selected_tab = st.sidebar.radio("Navigate", tab_labels)

st.sidebar.divider()
st.sidebar.caption(
    "**Model:** Log-log OLS elasticity (HC3 SE) · "
    "GBM demand forecast · Beta-Binomial A/B"
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with st.spinner("Loading data and fitting elasticity models..."):
    df       = load_data()
    elast_df = load_elasticities(df)

CLUSTERS        = sorted(df["cluster"].unique().tolist())
CLUSTER_COLORS  = {
    "europe_leisure":   "#2E86AB",
    "asia_budget":      "#A23B72",
    "americas_premium": "#F18F01",
    "mea_emerging":     "#C73E1D",
    "longhaul_exotic":  "#3B1F2B",
}

# ---------------------------------------------------------------------------
# TAB 1: Single Destination Recommender
# ---------------------------------------------------------------------------

if selected_tab == tab_labels[0]:
    st.title("📍 Single Destination Pricing")
    st.caption("Input destination parameters to get a pricing recommendation with full reasoning.")

    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.subheader("Inputs")

        dest_id = st.text_input("Destination ID", value="DEST_001")
        cluster = st.selectbox("Cluster", CLUSTERS, index=CLUSTERS.index("europe_leisure"))

        st.divider()
        st.markdown("**Pricing**")
        current_price = st.number_input("Current price (USD)", min_value=1.0, max_value=100.0,
                                         value=9.82, step=0.50)
        ref_price     = st.number_input("Reference price (USD)", min_value=1.0, max_value=100.0,
                                         value=9.82, step=0.50)
        comp_price    = st.number_input("Competitor price (USD)", min_value=1.0, max_value=100.0,
                                         value=9.50, step=0.50)
        cost_fraction = st.slider("Cost as % of ref price", 20, 60, 35) / 100.0
        cost_per_unit = ref_price * cost_fraction

        st.divider()
        st.markdown("**Demand**")
        current_cr        = st.slider("Current CR (%)", 0.5, 25.0, 5.6, step=0.1) / 100.0
        forecasted_sessions = st.number_input("Forecasted daily sessions", 10, 5000, 120)

        st.divider()
        st.markdown("**Elasticity** (from model or manual override)")
        model_elast = float(elast_df.loc[cluster, "elasticity"]) if cluster in elast_df.index else -2.0
        elasticity  = st.slider("Own-price elasticity (β)",
                                 min_value=-5.0, max_value=-0.5,
                                 value=round(model_elast, 2), step=0.05)
        cross_elast = st.slider("Cross-price elasticity (γ)",
                                 min_value=0.0, max_value=1.5,
                                 value=0.40, step=0.05)

        st.divider()
        st.markdown("**Constraints**")
        mode = st.selectbox("Objective", [
            "margin_optimal", "revenue_optimal", "constrained"
        ])
        max_price_change = st.slider("Max price change (%)", 5, 50, 30) / 100.0
        min_margin_pct   = st.slider("Min margin % of price", 5, 40, 20) / 100.0
        min_cr_pct       = st.slider("Min CR floor (%) — constrained mode only", 1, 20, 3) / 100.0

    with col2:
        inp = PricingInput(
            destination_id      = dest_id,
            cluster             = cluster,
            current_price       = current_price,
            ref_price           = ref_price,
            elasticity          = elasticity,
            cross_elasticity    = cross_elast,
            comp_price          = comp_price,
            cost_per_unit       = cost_per_unit,
            forecasted_sessions = float(forecasted_sessions),
            current_cr          = current_cr,
            min_margin_pct      = min_margin_pct,
            max_price_change    = max_price_change,
            min_cr              = min_cr_pct,
        )
        rec = recommend_price(inp, mode=mode)

        # --- Headline metrics ---
        st.subheader("Recommendation")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recommended Price",
                  f"${rec.recommended_price:.2f}",
                  f"{rec.price_change_pct:+.1%}")
        m2.metric("Expected CR",
                  f"{rec.recommended_cr:.2%}",
                  f"{rec.cr_change_pct:+.1%}")
        m3.metric("Margin / Session",
                  f"${rec.recommended_margin_per_session:.4f}",
                  f"{rec.margin_uplift_pct:+.1%}")
        m4.metric("Daily Margin",
                  f"${rec.recommended_daily_margin:.0f}",
                  f"${rec.daily_margin_uplift:+.0f}")

        st.divider()

        # --- Pareto frontier plot ---
        st.subheader("CR–Margin Pareto Frontier")
        frontier = compute_pareto_frontier(
            base_price    = current_price,
            ref_price     = ref_price,
            elasticity    = elasticity,
            base_cr       = current_cr,
            cost_per_unit = cost_per_unit,
            comp_price    = comp_price,
            cross_elast   = cross_elast,
            price_range   = (0.5, 1.5),
        )

        color = CLUSTER_COLORS.get(cluster, "#2E86AB")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

        # Left: CR vs Margin per session
        axes[0].plot(frontier["cr"]*100, frontier["margin_per_session"],
                     color=color, lw=2)
        axes[0].scatter([rec.current_cr*100], [rec.current_margin_per_session],
                        color="grey", s=100, zorder=5, label="Current", marker="o")
        axes[0].scatter([rec.recommended_cr*100], [rec.recommended_margin_per_session],
                        color=color, s=150, zorder=6, label="Recommended", marker="*")
        axes[0].set_xlabel("Conversion Rate (%)")
        axes[0].set_ylabel("Margin per Session (USD)")
        axes[0].set_title("CR–Margin Frontier")
        axes[0].legend(fontsize=8)
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        # Right: Margin per session vs Price
        axes[1].plot(frontier["price"], frontier["margin_per_session"],
                     color=color, lw=2)
        axes[1].axvline(rec.current_price, color="grey", lw=1.5,
                        linestyle="--", label=f"Current ${current_price:.2f}")
        axes[1].axvline(rec.recommended_price, color=color, lw=2,
                        linestyle="--", label=f"Recommended ${rec.recommended_price:.2f}")
        axes[1].scatter([rec.recommended_price], [rec.recommended_margin_per_session],
                        color=color, s=150, zorder=6, marker="*")
        axes[1].set_xlabel("Price (USD)")
        axes[1].set_ylabel("Margin per Session (USD)")
        axes[1].set_title("Margin vs Price")
        axes[1].legend(fontsize=8)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # --- Reasoning ---
        st.subheader("Reasoning")
        for line in rec.reasoning:
            st.markdown(f"- {line}")

        if rec.constraint_active:
            st.warning("⚠️ Min CR floor is active — unconstrained optimum would price lower.")

# ---------------------------------------------------------------------------
# TAB 2: Portfolio Overview
# ---------------------------------------------------------------------------

elif selected_tab == tab_labels[1]:
    st.title("🗺️ Portfolio Pricing Overview")
    st.caption("Batch recommendations across all 50 destinations.")

    col_l, col_r = st.columns([1, 3])
    with col_l:
        batch_mode = st.selectbox("Objective", ["margin_optimal", "revenue_optimal", "constrained"])
        run_batch  = st.button("Run Recommendations", type="primary")

    if run_batch:
        with st.spinner("Running batch pricing engine..."):
            batch_df = batch_recommend(df, elast_df, None, mode=batch_mode)

        st.subheader("Summary by Cluster")
        cluster_summary = (
            batch_df.groupby("cluster")
            .agg(
                n_destinations   = ("destination_id", "count"),
                avg_price_change = ("price_change_pct", "mean"),
                avg_cr_change    = ("cr_change_pct", "mean"),
                avg_margin_uplift= ("margin_uplift_pct", "mean"),
                total_daily_uplift=("daily_margin_uplift", "sum"),
            )
            .sort_values("total_daily_uplift", ascending=False)
        )
        st.dataframe(
            cluster_summary.style.format({
                "avg_price_change":  "{:+.1%}",
                "avg_cr_change":     "{:+.1%}",
                "avg_margin_uplift": "{:+.1%}",
                "total_daily_uplift":"${:+.0f}",
            }).background_gradient(subset=["avg_margin_uplift"], cmap="RdYlGn"),
            use_container_width=True,
        )

        total_uplift = batch_df["daily_margin_uplift"].sum()
        st.metric("Total Portfolio Daily Margin Uplift",
                  f"${total_uplift:+.0f}/day",
                  f"${total_uplift*365:+,.0f} annualised")

        st.subheader("All Destinations")
        display_cols = ["destination_id","cluster","current_price","recommended_price",
                        "price_change_pct","current_cr","recommended_cr",
                        "cr_change_pct","margin_uplift_pct","daily_margin_uplift"]
        st.dataframe(
            batch_df[display_cols].style.format({
                "current_price":      "${:.2f}",
                "recommended_price":  "${:.2f}",
                "price_change_pct":   "{:+.1%}",
                "current_cr":         "{:.2%}",
                "recommended_cr":     "{:.2%}",
                "cr_change_pct":      "{:+.1%}",
                "margin_uplift_pct":  "{:+.1%}",
                "daily_margin_uplift":"${:+.0f}",
            }).background_gradient(subset=["margin_uplift_pct"], cmap="RdYlGn"),
            use_container_width=True,
            height=500,
        )

        # Scatter: price change vs margin uplift
        fig, ax = plt.subplots(figsize=(8, 4))
        for cluster, grp in batch_df.groupby("cluster"):
            ax.scatter(
                grp["price_change_pct"] * 100,
                grp["margin_uplift_pct"] * 100,
                label=cluster, color=CLUSTER_COLORS.get(cluster, "grey"),
                s=60, alpha=0.85
            )
        ax.axhline(0, color="grey", lw=0.8, linestyle="--")
        ax.axvline(0, color="grey", lw=0.8, linestyle="--")
        ax.set_xlabel("Recommended Price Change (%)")
        ax.set_ylabel("Margin per Session Uplift (%)")
        ax.set_title("Price Change vs Margin Uplift by Destination")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    else:
        st.info("Click **Run Recommendations** to generate batch pricing suggestions for all 50 destinations.")

# ---------------------------------------------------------------------------
# TAB 3: Experiment Designer
# ---------------------------------------------------------------------------

elif selected_tab == tab_labels[2]:
    st.title("🧪 A/B Experiment Designer")
    st.caption("Power analysis calculator for pricing experiments.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Parameters")
        baseline_cr    = st.slider("Baseline CR (%)", 1.0, 20.0, 5.6, 0.1) / 100.0
        daily_sessions = st.number_input("Daily sessions (cluster-level)", 50, 10000, 1680)
        mde_input      = st.slider("Target MDE (% relative CR lift)", 5, 50, 10)
        alpha          = st.selectbox("Significance level (α)", [0.05, 0.01, 0.10], index=0)
        power          = st.selectbox("Power (1-β)", [0.80, 0.90, 0.95], index=0)

        n_required = sample_size_per_variant(baseline_cr, mde_input/100, alpha, power)
        duration   = experiment_duration_days(n_required, daily_sessions)
        n_30_days  = int(daily_sessions * 30 / 2)
        mde_30     = mde_from_sample_size(n_30_days, baseline_cr, alpha, power)

        st.divider()
        st.subheader("Results")
        r1, r2 = st.columns(2)
        r1.metric("Sessions per Variant", f"{n_required:,}")
        r2.metric("Experiment Duration", f"{duration} days ({duration/7:.1f}w)")
        r1.metric("MDE achievable in 30 days", f"{mde_30:.1%}")
        r2.metric("Total sessions needed", f"{n_required*2:,}")

    with col2:
        st.subheader("Design Table")
        summary = design_summary(baseline_cr, daily_sessions, alpha=alpha, power=power)
        st.dataframe(
            summary.style.format({
                "mde_relative":   "{:.0%}",
                "mde_absolute":   "{:.4f}",
                "n_per_variant":  "{:,}",
                "total_sessions": "{:,}",
                "duration_days":  "{:,}",
                "duration_weeks": "{:.1f}",
            }).background_gradient(subset=["duration_days"], cmap="RdYlGn_r"),
            use_container_width=True,
        )

        st.subheader("Precision–Duration Trade-off")
        mde_range  = np.linspace(0.05, 0.50, 80)
        day_values = [experiment_duration_days(
            sample_size_per_variant(baseline_cr, m, alpha, power),
            daily_sessions
        ) for m in mde_range]

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(mde_range * 100, day_values, color="#2E86AB", lw=2)
        ax.axvline(mde_input, color="#C73E1D", lw=1.5, linestyle="--",
                   label=f"Your MDE = {mde_input}% → {duration}d")
        ax.axhline(30, color="grey", lw=1, linestyle=":", alpha=0.7, label="30-day budget")
        ax.scatter([mde_input], [duration], color="#C73E1D", s=80, zorder=5)
        ax.set_xlabel("MDE (% relative CR lift)")
        ax.set_ylabel("Experiment Duration (days)")
        ax.set_title("Duration vs MDE")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.subheader("⚠️ The Peeking Problem")
    st.markdown(
        "Checking significance daily and stopping at p < 0.05 inflates your false positive rate "
        "from the nominal 5% to ~15–20%. This is because repeated testing on accumulating data "
        "is effectively running multiple tests on the same experiment.\n\n"
        "**Solutions:**\n"
        "- Check significance only at the pre-specified end date\n"
        "- Use sequential testing (always-valid p-values)\n"
        "- Use the Bayesian framework (Hito 4 notebook) which supports continuous monitoring"
    )
