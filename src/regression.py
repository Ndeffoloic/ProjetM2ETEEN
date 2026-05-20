"""
High-frequency OLS regression module.

Single Responsibility: runs regressions of energy price variables on the news index.
Open/Closed: accepts real FRED data or falls back to synthetic.

Main regression (per feedback):
    ratio_return_t = alpha + beta * index_score_t + epsilon_t

Where ratio_return = pct_change(electricity_price / oil_price).
A negative beta means subsidies lower electricity relative to oil (expected effect).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import log


def run_regression(
    index_df: pd.DataFrame,
    energy_df: pd.DataFrame | None = None,
    target_col: str = "ratio_return",
) -> tuple[dict, pd.DataFrame]:
    """
    OLS: target_t = alpha + beta * index_score_t + epsilon_t

    Parameters:
        index_df: weekly news index (must have 'week_start', 'index_score')
        energy_df: FRED energy prices (from fred_data.fetch_energy_prices)
        target_col: which column to regress on ('ratio_return', 'elec_return', 'oil_return')

    Returns (results_dict, merged_dataframe).
    """
    df = index_df.copy()

    # Normalise week_start on both sides:
    #   1. Strip timezone (UTC vs naive mismatch)
    #   2. Strip time component (12:34:30 vs 00:00:00 mismatch from NYT article times)
    if df["week_start"].dt.tz is not None:
        df["week_start"] = df["week_start"].dt.tz_localize(None)
    df["week_start"] = df["week_start"].dt.normalize()  # → 00:00:00

    if energy_df is not None and not energy_df.empty:
        energy = energy_df.copy()
        if energy["week_start"].dt.tz is not None:
            energy["week_start"] = energy["week_start"].dt.tz_localize(None)
        energy["week_start"] = energy["week_start"].dt.normalize()
        # Merge on week_start
        df = df.merge(energy, on="week_start", how="inner")
        log.info(f"Merged index with FRED data: {len(df)} weeks overlap")

        if target_col not in df.columns:
            log.error(f"Target column '{target_col}' not found. Available: {list(df.columns)}")
            return {"error": f"missing_column_{target_col}"}, df
    else:
        log.warning("No FRED data — using synthetic returns for demo.")
        rng = np.random.default_rng(42)
        n = len(df)
        df["ratio_return"] = -0.003 * df["index_score"] + rng.normal(0, 0.02, n)
        df["oil_return"] = -0.005 * df["index_score"] + rng.normal(0, 0.03, n)
        df["elec_return"] = -0.001 * df["index_score"] + rng.normal(0, 0.015, n)

    # Drop NaN/inf
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["index_score", target_col])

    if len(df) < 10:
        log.error(f"Insufficient data for regression: {len(df)} obs.")
        return {"error": "insufficient_data", "n_obs": len(df)}, df

    X = sm.add_constant(df["index_score"])
    y = df[target_col]
    model = sm.OLS(y, X).fit(cov_type="HC1")

    log.info(f"\n{model.summary()}")

    beta = model.params.get("index_score", np.nan)

    # Interpretation depends on target AND significance
    # CRITICAL: Do NOT interpret when p > 0.10 (even p > 0.05 is weak)
    pval = model.pvalues.get("index_score", 1.0)
    is_significant = pval < 0.10

    if not is_significant:
        # No detectable effect. Be honest.
        interpretation = (
            f"Aucun effet detectable des news de subventions sur {target_col} "
            f"(p = {pval:.3f}, bien au-delà du seuil alpha=0.10). "
            f"Les prix/ratios d'energie ne repondent pas mesurablement aux annonces "
            f"climat du NYT à l'horizon hebdomadaire. Ce résultat est cohérent avec "
            f"l'integralité des marchés énergétiques (OPEP, géopolitique, demande globale "
            f"écrasent les signaux médiatiques locaux)."
        )
    elif target_col == "ratio_return":
        interpretation = (
            "Les subventions reduisent le cout relatif de l'electricite vs. petrole (substitution)"
            if beta < 0
            else "Les subventions augmentent le ratio elec/petrole (effet inattendu ou fossilflation)"
        )
    elif target_col == "ratio_gas_return":
        interpretation = (
            "Les subventions reduisent le cout relatif elec/gaz "
            "(gaz = combustible marginal de l'electricite US, canal de substitution direct)"
            if beta < 0
            else "Les subventions augmentent le ratio elec/gaz (anti-substitution, inattendu)"
        )
    elif target_col == "oil_return":
        interpretation = (
            "Green Paradox (extraction acceleree)"
            if beta < 0
            else "Fossilflation (retention d'offre)"
        )
    else:
        interpretation = f"beta {'<' if beta < 0 else '>'} 0 sur {target_col}"

    # ----------------------------------------------------------------
    # Economic magnitude (per reviewer feedback)
    # ----------------------------------------------------------------
    # target is a weekly return (pct_change), so beta * S_t = % change in week t.
    # Translate into tangible scenarios:
    sd_index = df["index_score"].std()
    sd_target = df[target_col].std()
    mean_target_abs = df[target_col].abs().mean()

    # 1) Effect of a 1-SD positive surprise in the news index
    effect_1sd_pct = beta * sd_index * 100  # in %

    # 2) Effect of an IRA-magnitude shock (direction = +1, importance = 3 → score = 3)
    effect_ira_pct = beta * 3 * 100  # in %

    # 3) Share of weekly target volatility explained per unit of index
    signal_to_noise = abs(beta * sd_index) / sd_target if sd_target > 0 else np.nan

    # 4) Annualized cumulative effect of a sustained +1 score every week (52 weeks)
    #    Compounded: (1 + beta*1)^52 - 1
    try:
        effect_annualized_pct = ((1 + beta) ** 52 - 1) * 100
    except Exception:
        effect_annualized_pct = np.nan

    results = {
        "target": target_col,
        "alpha": model.params.get("const", np.nan),
        "beta": beta,
        "beta_se": model.bse.get("index_score", np.nan),
        "beta_tstat": model.tvalues.get("index_score", np.nan),
        "beta_pvalue": model.pvalues.get("index_score", np.nan),
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "n_obs": int(model.nobs),
        "f_stat": model.fvalue,
        "interpretation": interpretation,
        # --- Economic magnitude metrics ---
        "sd_index": sd_index,
        "sd_target": sd_target,
        "mean_target_abs": mean_target_abs,
        "effect_1sd_pct": effect_1sd_pct,
        "effect_ira_pct": effect_ira_pct,
        "signal_to_noise": signal_to_noise,
        "effect_annualized_pct": effect_annualized_pct,
    }

    return results, df
