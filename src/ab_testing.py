"""
esim-pricing-engine / src/ab_testing.py
=========================================
Reusable A/B testing framework for pricing experiments.

Two paradigms implemented
--------------------------
1. Frequentist  — classical power analysis + two-proportion z-test
   Use when: regulatory reporting, strict Type I error control required
   Output: p-value, reject/fail-to-reject at α=0.05

2. Bayesian (Beta-Binomial) — posterior inference on CR difference
   Use when: sequential monitoring, early stopping, probabilistic decisions
   Output: P(B > A), posterior distributions, credible intervals

Experiment design utilities
----------------------------
- sample_size_per_variant(): given MDE and baseline CR, how many sessions needed?
- experiment_duration_days(): given daily traffic, how many days to run?
- mde_from_sample_size(): given n, what's the smallest detectable effect?

These are the tools you use BEFORE running an experiment.
The evaluators are what you use AFTER.

References
----------
- Evan Miller: "How Not To Run an A/B Test" (peeking problem)
- VWO Bayesian A/B testing: Beta-Binomial conjugate model
- Deng et al. (2013): "Improving the Sensitivity of Online Controlled Experiments"
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import betaln
from typing import Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Experiment Design (pre-experiment)
# ---------------------------------------------------------------------------

def sample_size_per_variant(
    baseline_cr: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> int:
    """
    Minimum sample size per variant for a CR experiment.

    Uses the standard two-proportion z-test formula.

    Parameters
    ----------
    baseline_cr   : current conversion rate (e.g. 0.045 for 4.5%)
    mde_relative  : minimum detectable effect as relative change (e.g. 0.10 for +10% CR)
    alpha         : Type I error rate (default 0.05)
    power         : 1 - Type II error rate (default 0.80)
    two_sided     : whether to use two-sided test (default True)

    Returns
    -------
    int: minimum sessions per variant
    """
    p1 = baseline_cr
    p2 = baseline_cr * (1 + mde_relative)

    sides = 2 if two_sided else 1
    z_alpha = stats.norm.ppf(1 - alpha / sides)
    z_beta  = stats.norm.ppf(power)

    # Pooled proportion under H0
    p_bar = (p1 + p2) / 2

    numerator   = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
                   z_beta  * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominator = (p2 - p1) ** 2

    return int(np.ceil(numerator / denominator))


def experiment_duration_days(
    n_per_variant: int,
    daily_sessions: int,
    n_variants: int = 2,
    traffic_split: float = 0.50,
) -> int:
    """
    How many days to run the experiment given daily traffic.

    Parameters
    ----------
    n_per_variant  : required sessions per variant (from sample_size_per_variant)
    daily_sessions : total daily sessions for the destination/segment
    n_variants     : number of variants (default 2: control + treatment)
    traffic_split  : fraction of traffic allocated to experiment (default 0.50)
                     (the other 50% keeps seeing the current price)

    Returns
    -------
    int: minimum experiment duration in days
    """
    sessions_per_variant_per_day = (daily_sessions * traffic_split) / n_variants
    return int(np.ceil(n_per_variant / sessions_per_variant_per_day))


def mde_from_sample_size(
    n_per_variant: int,
    baseline_cr: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
    precision: float = 0.0001,
) -> float:
    """
    Inverse of sample_size_per_variant: given n, what's the minimum detectable
    relative effect?

    Uses binary search over mde_relative.
    """
    lo, hi = 0.001, 5.0  # search range: 0.1% to 500% relative change
    while hi - lo > precision:
        mid = (lo + hi) / 2
        if sample_size_per_variant(baseline_cr, mid, alpha, power, two_sided) <= n_per_variant:
            hi = mid
        else:
            lo = mid
    return round(hi, 6)


def design_summary(
    baseline_cr: float,
    daily_sessions: int,
    mde_relative: float = 0.10,
    alpha: float = 0.05,
    power: float = 0.80,
) -> pd.DataFrame:
    """
    Produce a design summary table for a range of MDE values.
    Useful for communicating the precision/duration trade-off to stakeholders.
    """
    mde_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    rows = []
    for mde in mde_values:
        n   = sample_size_per_variant(baseline_cr, mde, alpha, power)
        days = experiment_duration_days(n, daily_sessions)
        rows.append({
            "mde_relative":         mde,
            "mde_absolute":         round(baseline_cr * mde, 5),
            "n_per_variant":        n,
            "total_sessions":       n * 2,
            "duration_days":        days,
            "duration_weeks":       round(days / 7, 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Frequentist Evaluator (post-experiment)
# ---------------------------------------------------------------------------

def frequentist_test(
    n_control: int,
    conversions_control: int,
    n_treatment: int,
    conversions_treatment: int,
    alpha: float = 0.05,
) -> dict:
    """
    Two-proportion z-test for CR difference.

    Parameters
    ----------
    n_control            : sessions in control group
    conversions_control  : transactions in control group
    n_treatment          : sessions in treatment group
    conversions_treatment: transactions in treatment group
    alpha                : significance level

    Returns
    -------
    dict with: cr_control, cr_treatment, relative_lift, z_stat, p_value,
               significant, confidence_interval_95
    """
    cr_a = conversions_control  / n_control
    cr_b = conversions_treatment / n_treatment

    # Pooled proportion
    p_pool = (conversions_control + conversions_treatment) / (n_control + n_treatment)
    se     = np.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_treatment))

    z_stat = (cr_b - cr_a) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # two-sided

    # 95% CI on the difference
    se_diff = np.sqrt(cr_a*(1-cr_a)/n_control + cr_b*(1-cr_b)/n_treatment)
    z_crit  = stats.norm.ppf(1 - alpha/2)
    ci_lo   = (cr_b - cr_a) - z_crit * se_diff
    ci_hi   = (cr_b - cr_a) + z_crit * se_diff

    return {
        "cr_control":           round(cr_a, 6),
        "cr_treatment":         round(cr_b, 6),
        "absolute_lift":        round(cr_b - cr_a, 6),
        "relative_lift":        round((cr_b - cr_a) / cr_a, 4),
        "z_stat":               round(z_stat, 4),
        "p_value":              round(p_value, 6),
        "significant":          p_value < alpha,
        "ci_95_lo":             round(ci_lo, 6),
        "ci_95_hi":             round(ci_hi, 6),
        "alpha":                alpha,
    }


# ---------------------------------------------------------------------------
# Bayesian Evaluator — Beta-Binomial conjugate model
# ---------------------------------------------------------------------------

def bayesian_test(
    n_control: int,
    conversions_control: int,
    n_treatment: int,
    conversions_treatment: int,
    prior_alpha: float = 1.0,
    prior_beta: float  = 1.0,
    n_samples: int     = 100_000,
    rope_lo: float     = -0.005,
    rope_hi: float     =  0.005,
) -> dict:
    """
    Bayesian A/B test using Beta-Binomial conjugate model.

    Prior: Beta(prior_alpha, prior_beta) — default is uniform (non-informative).
    Posterior: Beta(alpha + conversions, beta + non-conversions)

    Parameters
    ----------
    rope_lo, rope_hi : Region Of Practical Equivalence — a difference smaller
                       than this is considered practically zero. Default ±0.5pp.

    Returns
    -------
    dict with: prob_treatment_better, prob_rope, posterior_means,
               credible_interval_95, expected_loss
    """
    rng = np.random.default_rng(42)

    # Posterior parameters
    alpha_a = prior_alpha + conversions_control
    beta_a  = prior_beta  + (n_control - conversions_control)
    alpha_b = prior_alpha + conversions_treatment
    beta_b  = prior_beta  + (n_treatment - conversions_treatment)

    # Sample from posteriors
    samples_a = rng.beta(alpha_a, beta_a, size=n_samples)
    samples_b = rng.beta(alpha_b, beta_b, size=n_samples)
    diff      = samples_b - samples_a

    # P(B > A)
    prob_b_better = np.mean(diff > 0)

    # P(practically equivalent) — difference falls in ROPE
    prob_rope = np.mean((diff >= rope_lo) & (diff <= rope_hi))

    # Expected loss (cost of choosing B if A is actually better)
    expected_loss_b = np.mean(np.maximum(samples_a - samples_b, 0))
    expected_loss_a = np.mean(np.maximum(samples_b - samples_a, 0))

    # 95% Credible Interval on the difference
    ci_lo, ci_hi = np.percentile(diff, [2.5, 97.5])

    return {
        "posterior_mean_control":   round(alpha_a / (alpha_a + beta_a), 6),
        "posterior_mean_treatment": round(alpha_b / (alpha_b + beta_b), 6),
        "prob_treatment_better":    round(prob_b_better, 4),
        "prob_rope":                round(prob_rope, 4),
        "expected_loss_if_launch":  round(expected_loss_b, 6),
        "expected_loss_if_rollback":round(expected_loss_a, 6),
        "ci_95_lo":                 round(ci_lo, 6),
        "ci_95_hi":                 round(ci_hi, 6),
        "posterior_samples_a":      samples_a,   # for plotting
        "posterior_samples_b":      samples_b,
        "diff_samples":             diff,
    }


# ---------------------------------------------------------------------------
# Simulate an A/B experiment (for notebook demo)
# ---------------------------------------------------------------------------

def simulate_experiment(
    baseline_cr: float,
    true_lift_relative: float,
    n_per_variant: int,
    daily_sessions: int,
    seed: int = 42,
) -> Tuple[dict, dict]:
    """
    Simulate a completed pricing A/B experiment.

    Returns (experiment_data, ground_truth) where experiment_data contains
    the observed counts and ground_truth contains the true effect.
    """
    rng = np.random.default_rng(seed)

    true_cr_control   = baseline_cr
    true_cr_treatment = baseline_cr * (1 + true_lift_relative)

    conversions_a = int(rng.binomial(n_per_variant, true_cr_control))
    conversions_b = int(rng.binomial(n_per_variant, true_cr_treatment))

    duration = experiment_duration_days(n_per_variant, daily_sessions)

    experiment_data = {
        "n_control":             n_per_variant,
        "conversions_control":   conversions_a,
        "n_treatment":           n_per_variant,
        "conversions_treatment": conversions_b,
        "observed_cr_control":   conversions_a / n_per_variant,
        "observed_cr_treatment": conversions_b / n_per_variant,
        "duration_days":         duration,
    }
    ground_truth = {
        "true_cr_control":   true_cr_control,
        "true_cr_treatment": true_cr_treatment,
        "true_lift_relative": true_lift_relative,
    }
    return experiment_data, ground_truth


# ---------------------------------------------------------------------------
# Peeking simulation — illustrate the peeking problem
# ---------------------------------------------------------------------------

def simulate_peeking(
    baseline_cr: float,
    n_total: int,
    true_lift: float = 0.0,   # null hypothesis: no effect
    n_simulations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate the peeking problem: if you check significance every day
    and stop when p < alpha, what's your actual Type I error rate?

    Returns DataFrame with one row per simulation, showing whether
    a false positive was declared.
    """
    rng  = np.random.default_rng(seed)
    cr_a = baseline_cr
    cr_b = baseline_cr * (1 + true_lift)

    results = []
    check_points = [int(n_total * f) for f in [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]]

    for sim in range(n_simulations):
        obs_a = rng.binomial(1, cr_a, n_total)
        obs_b = rng.binomial(1, cr_b, n_total)

        peeked_significant = False
        final_significant  = False

        for cp in check_points:
            res = frequentist_test(cp, obs_a[:cp].sum(), cp, obs_b[:cp].sum(), alpha)
            if res["significant"] and not peeked_significant:
                peeked_significant = True
            if cp == n_total:
                final_significant = res["significant"]

        results.append({
            "peeked_significant": peeked_significant,
            "final_significant":  final_significant,
        })

    df = pd.DataFrame(results)
    return df
