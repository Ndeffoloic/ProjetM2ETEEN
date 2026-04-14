"""
Main async pipeline orchestrator.

Liskov Substitution: each stage function can be swapped independently.
Single Responsibility: only orchestrates the pipeline flow.
"""

import asyncio
import sys

import pandas as pd

from .config import PipelineConfig, log
from .scraper import scrape_nyt
from .scorer import score_articles
from .index_builder import build_weekly_index
from .regression import run_regression


async def run_pipeline(cfg: PipelineConfig | None = None) -> dict:
    """
    Execute the full 5-stage pipeline.
    Returns a dict with all intermediate results for Streamlit consumption.
    """
    if cfg is None:
        cfg = PipelineConfig()

    log.info("=" * 70)
    log.info("NYT Green Subsidy Index Pipeline — Acharya et al. (2025)")
    log.info("=" * 70)

    # Stage 1: Scrape
    log.info("[1/4] Scraping NYT Archive API...")
    df_raw = await scrape_nyt(cfg)
    if df_raw.empty:
        log.error("No articles retrieved. Check API key and network.")
        return {"error": "no_articles"}
    df_raw.to_csv(cfg.raw_csv, index=False)

    # Stage 2: LLM Scoring
    log.info("[2/4] Scoring articles via Local AI Stack...")
    df_scored = await score_articles(df_raw, cfg)
    df_scored.to_csv(cfg.raw_csv, index=False)

    # Stage 3: Weekly Index
    log.info("[3/4] Building weekly index...")
    weekly = build_weekly_index(df_scored)
    weekly.to_csv(cfg.index_csv, index=False)

    # Stage 4: Regression
    log.info("[4/4] Running high-frequency regression...")
    reg_results, weekly_aug = run_regression(weekly)
    pd.DataFrame([reg_results]).to_csv(cfg.regression_csv, index=False)

    log.info("=" * 70)
    log.info("PIPELINE COMPLETE")
    log.info(f"  Articles scraped:  {len(df_raw)}")
    log.info(f"  LLM-relevant:     {(df_scored['relevance'] == 1).sum()}")
    log.info(f"  Index weeks:       {len(weekly)}")
    log.info(f"  beta (subsidy->oil): {reg_results.get('beta', 'N/A')}")
    log.info("=" * 70)

    return {
        "raw": df_raw,
        "scored": df_scored,
        "weekly": weekly,
        "weekly_aug": weekly_aug,
        "regression": reg_results,
        "config": cfg,
    }


def main():
    """CLI entry point."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
