"""
Weekly Green Subsidy News Index construction.

Single Responsibility: aggregates scored articles into a weekly time series.
Formula: Index_w = sum(direction_i x importance_i) for relevant articles in week w.
"""

import pandas as pd

from .config import log


def build_weekly_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the weekly index from LLM-scored articles.

    Returns a DataFrame with columns:
        year_week, week_start, index_score, n_articles, avg_importance
    Missing weeks are filled with 0.
    """
    relevant = df[df["relevance"] == 1].copy()

    if relevant.empty:
        log.warning("No relevant articles — index will be empty.")
        return pd.DataFrame(columns=["year_week", "week_start", "index_score", "n_articles", "avg_importance"])

    relevant["score"] = relevant["direction"] * relevant["importance"]

    # ISO year-week (avoids cross-year collision)
    relevant["iso_year"] = relevant["date"].dt.isocalendar().year.astype(int)
    relevant["iso_week"] = relevant["date"].dt.isocalendar().week.astype(int)
    relevant["year_week"] = (
        relevant["iso_year"].astype(str) + "-W" + relevant["iso_week"].astype(str).str.zfill(2)
    )

    weekly = (
        relevant
        .groupby("year_week")
        .agg(
            index_score=("score", "sum"),
            n_articles=("score", "count"),
            avg_importance=("importance", "mean"),
            week_start=("date", "min"),
        )
        .reset_index()
        .sort_values("week_start")
        .reset_index(drop=True)
    )

    # Fill missing weeks with 0
    if not weekly.empty:
        full_range = pd.date_range(weekly["week_start"].min(), weekly["week_start"].max(), freq="W-MON")
        full_weeks = pd.DataFrame({"week_start": full_range})
        full_weeks["year_week"] = (
            full_weeks["week_start"].dt.isocalendar().year.astype(str)
            + "-W"
            + full_weeks["week_start"].dt.isocalendar().week.astype(str).str.zfill(2)
        )
        weekly = full_weeks.merge(weekly.drop(columns=["week_start"]), on="year_week", how="left")
        weekly["index_score"] = weekly["index_score"].fillna(0)
        weekly["n_articles"] = weekly["n_articles"].fillna(0).astype(int)
        weekly["avg_importance"] = weekly["avg_importance"].fillna(0)

    log.info(f"Weekly index: {len(weekly)} weeks, non-zero: {(weekly['index_score'] != 0).sum()}")
    return weekly
