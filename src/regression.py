"""
High-frequency OLS regression module.

Single Responsibility: runs DP_oil = alpha + beta * Index + epsilon.
Open/Closed: accepts real or synthetic oil data without code changes.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import log


def run_regression(
    index_df: pd.DataFrame,
    oil_prices: pd.DataFrame | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    OLS: oil_return_w = alpha + beta * index_score_w + epsilon_w

    If oil_prices is None, generates synthetic WTI-like returns for demo.
    Returns (results_dict, augmented_dataframe).
    """
    df = index_df.copy()

    if oil_prices is None:
        log.warning("No real oil data — using synthetic WTI returns for demo.")
        rng = np.random.default_rng(42)
        n = len(df)
        synthetic_beta = -0.005  # Green Paradox baseline
        df["oil_return"] = synthetic_beta * df["index_score"] + rng.normal(0, 0.03, n)
    else:
        df = df.merge(oil_prices, on="week_start", how="inner")

    if len(df) < 10:
        log.error("Insufficient data for regression.")
        return {"error": "insufficient_data", "n_obs": len(df)}, df

    X = sm.add_constant(df["index_score"])
    y = df["oil_return"]
    model = sm.OLS(y, X).fit(cov_type="HC1")  # Heteroskedasticity-robust SEs

    log.info(f"\n{model.summary()}")

    beta = model.params.get("index_score", np.nan)
    results = {
        "alpha": model.params.get("const", np.nan),
        "beta": beta,
        "beta_se": model.bse.get("index_score", np.nan),
        "beta_tstat": model.tvalues.get("index_score", np.nan),
        "beta_pvalue": model.pvalues.get("index_score", np.nan),
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "n_obs": int(model.nobs),
        "f_stat": model.fvalue,
        "interpretation": (
            "Green Paradox (extraction acceleration)"
            if beta < 0
            else "Fossilflation (investment freeze -> supply hoarding)"
        ),
    }

    return results, df
