"""
=============================================================================
NYT Green Subsidy Index — Full Pipeline
=============================================================================
Acharya et al. (2025) Reproduction & Extension
Part 4: Empirical Exploration of Green Subsidies

Pipeline stages:
    1. Async scraping of NYT Archive API (rate-limited to 5 req/min)
    2. Section + keyword pre-filtering (word-boundary safe)
    3. LLM-based sentiment scoring via Local AI Stack (semaphore-gated)
    4. Weekly index construction (direction × importance aggregation)
    5. High-frequency regression against energy futures returns
    6. Diagnostic visualizations

Hardware constraint: AMD APU with ~8 GB shared VRAM
    → MAX_CONCURRENT_LLM_REQUESTS = 3 (asyncio.Semaphore)

Author: Loïc NEMBOT
=============================================================================
"""

import asyncio
import json
import re
import logging
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from openai import AsyncOpenAI

# Optional: tqdm for progress bars
try:
    from tqdm.asyncio import tqdm as async_tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Local AI Stack ---
LOCAL_AI_BASE_URL = "http://127.0.0.1:9485/v1"
LOCAL_MODEL_NAME  = "local-model"       # Adjust to match your n8n/Docker model name

# --- NYT API ---
NYT_API_KEY = "VOTRE_CLE_API_NYT_ICI"  # Replace with your real key
NYT_RATE_LIMIT_DELAY = 12.0            # seconds between Archive API calls (5 req/min)

# --- Hardware throttle ---
MAX_CONCURRENT_LLM_REQUESTS = 3        # Semaphore width for AMD 8 GB shared VRAM

# --- Scraping scope ---
SCRAPE_YEARS  = [2022, 2023, 2024]     # Configurable date range
SCRAPE_MONTHS = range(1, 13)

# --- Filtering ---
ALLOWED_SECTIONS = {
    "business", "climate", "energy", "national", "financial",
    "us", "world", "science", "your money",
}
# Word-boundary-safe keyword patterns (compiled once)
KEYWORD_PATTERNS = [
    re.compile(r"\bsubsid(?:y|ies|ize|ized|izing)\b", re.IGNORECASE),
    re.compile(r"\btax\s+credit\b", re.IGNORECASE),
    re.compile(r"\bgrant(?:s)?\b", re.IGNORECASE),
    re.compile(r"\brenewable(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bsolar\b", re.IGNORECASE),
    re.compile(r"\bwind\s+(?:energy|power|farm|turbine)\b", re.IGNORECASE),
    re.compile(r"\bgreen\s+energy\b", re.IGNORECASE),
    re.compile(r"\bclean\s+energy\b", re.IGNORECASE),
    re.compile(r"\binflation\s+reduction\s+act\b", re.IGNORECASE),
    re.compile(r"\bIRA\b"),  # Case-sensitive to avoid false positives
    re.compile(r"\bfeed[- ]in[- ]tariff\b", re.IGNORECASE),
    re.compile(r"\bEUA(?:s)?\b"),
]

# Require at least one "subsidy-type" keyword AND one "energy-type" keyword
SUBSIDY_PATTERNS = KEYWORD_PATTERNS[:3]   # subsidy, tax credit, grant
ENERGY_PATTERNS  = KEYWORD_PATTERNS[3:]   # renewable, solar, wind energy, etc.

# --- Output ---
OUTPUT_DIR = Path(".")
RAW_CSV       = OUTPUT_DIR / "nyt_articles_raw.csv"
INDEX_CSV     = OUTPUT_DIR / "weekly_green_subsidy_index.csv"
REGRESSION_CSV = OUTPUT_DIR / "regression_results.csv"
PLOT_INDEX    = OUTPUT_DIR / "weekly_index_plot.png"
PLOT_REGRESSION = OUTPUT_DIR / "regression_diagnostic.png"

# =============================================================================
# LLM CLIENT
# =============================================================================
client = AsyncOpenAI(
    base_url=LOCAL_AI_BASE_URL,
    api_key="not-needed-for-local",
)

# =============================================================================
# STAGE 1 — NYT ARCHIVE SCRAPING (rate-limited)
# =============================================================================

def passes_keyword_filter(text: str) -> bool:
    """
    Require at least one subsidy-related AND one energy-related keyword.
    This avoids false positives like 'wind down subsidies for coal'.
    """
    has_subsidy = any(p.search(text) for p in SUBSIDY_PATTERNS)
    has_energy  = any(p.search(text) for p in ENERGY_PATTERNS)
    return has_subsidy and has_energy


def passes_section_filter(article: dict) -> bool:
    """Check if the article belongs to an allowed section."""
    section = article.get("section_name", "").lower().strip()
    news_desk = article.get("news_desk", "").lower().strip()
    # Accept if either field matches
    return (section in ALLOWED_SECTIONS) or (news_desk in ALLOWED_SECTIONS)


async def fetch_nyt_month(
    session: aiohttp.ClientSession,
    year: int,
    month: int,
    rate_limiter: asyncio.Semaphore,
) -> list[dict]:
    """
    Fetch one month of NYT Archive data.
    Rate-limited via semaphore + sleep to respect 5 req/min.
    """
    async with rate_limiter:
        url = (
            f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"
            f"?api-key={NYT_API_KEY}"
        )
        log.info(f"Fetching NYT archive: {year}-{month:02d}")

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 429:
                    log.warning(f"Rate-limited on {year}-{month:02d}, retrying after 60s...")
                    await asyncio.sleep(60)
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as retry:
                        if retry.status != 200:
                            log.error(f"Retry failed for {year}-{month:02d}: HTTP {retry.status}")
                            return []
                        data = await retry.json()
                elif resp.status != 200:
                    log.error(f"HTTP {resp.status} for {year}-{month:02d}")
                    return []
                else:
                    data = await resp.json()
        except Exception as e:
            log.error(f"Network error for {year}-{month:02d}: {e}")
            return []

        articles = data.get("response", {}).get("docs", [])
        filtered = []

        for art in articles:
            # --- Section filter ---
            if not passes_section_filter(art):
                continue

            # --- Keyword filter (word-boundary safe) ---
            headline = art.get("headline", {}).get("main", "")
            snippet  = art.get("snippet", "") or ""
            lead     = art.get("lead_paragraph", "") or ""
            full_text = f"{headline} {snippet} {lead}"

            if not passes_keyword_filter(full_text):
                continue

            filtered.append({
                "date":       art.get("pub_date", ""),
                "headline":   headline,
                "snippet":    snippet,
                "lead_paragraph": lead,
                "section":    art.get("section_name", ""),
                "news_desk":  art.get("news_desk", ""),
                "web_url":    art.get("web_url", ""),
                "text_to_analyze": full_text,
            })

        log.info(f"  {year}-{month:02d}: {len(filtered)} relevant / {len(articles)} total")

        # Respect rate limit: wait before releasing semaphore
        await asyncio.sleep(NYT_RATE_LIMIT_DELAY)
        return filtered


async def scrape_all_nyt() -> pd.DataFrame:
    """Orchestrate the full NYT scraping with rate limiting."""
    # Semaphore = 1 ensures strictly sequential API calls (rate limit safety)
    rate_limiter = asyncio.Semaphore(1)
    all_articles = []

    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_nyt_month(session, year, month, rate_limiter)
            for year in SCRAPE_YEARS
            for month in SCRAPE_MONTHS
        ]
        # Sequential execution is forced by Semaphore(1), but gather handles ordering
        results = await asyncio.gather(*tasks)

    for batch in results:
        all_articles.extend(batch)

    df = pd.DataFrame(all_articles)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").reset_index(drop=True)

    log.info(f"Total articles after filtering: {len(df)}")
    return df


# =============================================================================
# STAGE 2 — LLM SCORING (semaphore-gated for AMD VRAM)
# =============================================================================

LLM_SYSTEM_PROMPT = """You are a financial climate-risk analyst specializing in energy transition policy.
You will receive a news snippet. Evaluate it and respond with ONLY a JSON object containing exactly these fields:

{
  "relevance": <0 or 1>,
  "direction": <-1, 0, or 1>,
  "importance": <1, 2, or 3>,
  "rationale": "<one sentence>"
}

Definitions:
- relevance: 1 if the article discusses government subsidies, tax credits, or grants for renewable/clean energy at a national or supranational level. 0 otherwise.
- direction: 1 if subsidies are being introduced, increased, or extended. -1 if they are being cut, repealed, or blocked. 0 if neutral, ambiguous, or merely descriptive.
- importance: 1 = minor (local grants, small programs). 2 = major (significant federal spending, e.g. multi-billion packages). 3 = massive / systemic (e.g. IRA, EU Green Deal-level legislation).
- rationale: one sentence explaining your scoring decision.

Output ONLY the JSON. No markdown, no commentary."""


def build_llm_prompt(article: dict) -> str:
    return (
        f"Headline: {article['headline']}\n"
        f"Snippet: {article['snippet']}\n"
        f"Lead: {article.get('lead_paragraph', '')}\n"
        f"Date: {article['date']}"
    )


async def score_article(
    article: dict,
    semaphore: asyncio.Semaphore,
    retry_count: int = 2,
) -> dict:
    """
    Score a single article via Local AI Stack.
    Semaphore ensures at most MAX_CONCURRENT_LLM_REQUESTS concurrent inferences.
    Includes retry logic for transient failures.
    """
    async with semaphore:
        prompt = build_llm_prompt(article)

        for attempt in range(retry_count + 1):
            try:
                response = await client.chat.completions.create(
                    model=LOCAL_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": LLM_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=256,
                )
                # BUG FIX: response.choices is a list → index [0]
                raw = response.choices[0].message.content.strip()

                # Robust JSON extraction (handle markdown code blocks)
                if raw.startswith("```"):
                    raw = re.sub(r"```(?:json)?\s*", "", raw).strip("`").strip()

                parsed = json.loads(raw)

                # Validate expected fields and ranges
                article["relevance"]  = int(parsed.get("relevance", 0)) if parsed.get("relevance") in (0, 1, "0", "1") else 0
                article["direction"]  = int(parsed.get("direction", 0))
                article["importance"] = int(parsed.get("importance", 1))
                article["rationale"]  = str(parsed.get("rationale", ""))

                # Clamp values to valid ranges
                article["direction"]  = max(-1, min(1, article["direction"]))
                article["importance"] = max(1, min(3, article["importance"]))

                return article

            except json.JSONDecodeError as e:
                log.warning(f"JSON parse error (attempt {attempt+1}): {e}")
                if attempt == retry_count:
                    article.update({"relevance": 0, "direction": 0, "importance": 0, "rationale": f"JSON_ERROR: {raw[:100]}"})
            except Exception as e:
                log.warning(f"LLM error (attempt {attempt+1}): {e}")
                if attempt == retry_count:
                    article.update({"relevance": 0, "direction": 0, "importance": 0, "rationale": f"LLM_ERROR: {str(e)[:100]}"})
                else:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

    return article


async def score_all_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Score all articles with hardware-gated concurrency."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_REQUESTS)
    articles = df.to_dict("records")

    log.info(f"Scoring {len(articles)} articles (max {MAX_CONCURRENT_LLM_REQUESTS} concurrent)...")

    tasks = [score_article(art, semaphore) for art in articles]

    if HAS_TQDM:
        scored = await async_tqdm.gather(*tasks, desc="LLM Scoring")
    else:
        scored = await asyncio.gather(*tasks)

    result = pd.DataFrame(scored)
    n_relevant = (result["relevance"] == 1).sum()
    log.info(f"Scoring complete: {n_relevant}/{len(result)} marked relevant by LLM")
    return result


# =============================================================================
# STAGE 3 — WEEKLY INDEX CONSTRUCTION
# =============================================================================

def build_weekly_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the Weekly Green Subsidy News Index.

    Index_w = sum( direction_i × importance_i ) for all relevant articles in week w

    This captures both the sign (pro-green vs anti-green) and the magnitude
    (minor grant vs IRA-level legislation) of each news event.
    """
    # Keep only LLM-confirmed relevant articles
    relevant = df[df["relevance"] == 1].copy()

    if relevant.empty:
        log.warning("No relevant articles found — index will be empty.")
        return pd.DataFrame(columns=["year_week", "index_score", "n_articles", "week_start"])

    # Compute article-level score
    relevant["score"] = relevant["direction"] * relevant["importance"]

    # ISO year-week (avoids cross-year week collision)
    relevant["iso_year"]  = relevant["date"].dt.isocalendar().year.astype(int)
    relevant["iso_week"]  = relevant["date"].dt.isocalendar().week.astype(int)
    relevant["year_week"] = relevant["iso_year"].astype(str) + "-W" + relevant["iso_week"].astype(str).str.zfill(2)

    # Aggregate: sum of scores + article count per week
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

    # Fill missing weeks with 0 (no news = no signal)
    if not weekly.empty:
        full_range = pd.date_range(
            weekly["week_start"].min(),
            weekly["week_start"].max(),
            freq="W-MON",
        )
        full_weeks = pd.DataFrame({"week_start": full_range})
        full_weeks["year_week"] = (
            full_weeks["week_start"].dt.isocalendar().year.astype(str)
            + "-W"
            + full_weeks["week_start"].dt.isocalendar().week.astype(str).str.zfill(2)
        )
        weekly = full_weeks.merge(weekly.drop(columns=["week_start"]), on="year_week", how="left")
        weekly["index_score"]    = weekly["index_score"].fillna(0)
        weekly["n_articles"]     = weekly["n_articles"].fillna(0).astype(int)
        weekly["avg_importance"] = weekly["avg_importance"].fillna(0)

    log.info(f"Weekly index: {len(weekly)} weeks, non-zero weeks: {(weekly['index_score'] != 0).sum()}")
    return weekly


# =============================================================================
# STAGE 4 — HIGH-FREQUENCY REGRESSION
# =============================================================================

def run_regression(
    index_df: pd.DataFrame,
    oil_prices: pd.DataFrame | None = None,
) -> dict:
    """
    OLS regression: ΔP_oil_w = α + β × Index_w + ε_w

    If no oil price data is supplied, generates synthetic WTI-like returns
    for demonstration / pipeline validation.

    Returns regression summary dict.
    """
    if oil_prices is None:
        log.warning("No real oil price data provided — using synthetic WTI returns for demo.")
        np.random.seed(42)
        n = len(index_df)
        # Synthetic weekly log-returns ~ N(0, 0.03) with small subsidy sensitivity
        synthetic_beta = -0.005  # Small negative: Green Paradox baseline
        noise = np.random.normal(0, 0.03, n)
        index_df = index_df.copy()
        index_df["oil_return"] = synthetic_beta * index_df["index_score"] + noise
    else:
        # Merge on week_start date (user must provide weekly oil returns)
        index_df = index_df.merge(oil_prices, on="week_start", how="inner")

    # Drop weeks where index is exactly 0 (no information) — optional
    # reg_data = index_df[index_df["index_score"] != 0].copy()
    reg_data = index_df.copy()

    if len(reg_data) < 10:
        log.error("Insufficient data for regression.")
        return {"error": "insufficient_data", "n_obs": len(reg_data)}

    X = sm.add_constant(reg_data["index_score"])
    y = reg_data["oil_return"]

    model = sm.OLS(y, X).fit(cov_type="HC1")  # Heteroskedasticity-robust SEs

    log.info(f"\n{model.summary()}")

    results = {
        "alpha":       model.params.get("const", np.nan),
        "beta":        model.params.get("index_score", np.nan),
        "beta_se":     model.bse.get("index_score", np.nan),
        "beta_tstat":  model.tvalues.get("index_score", np.nan),
        "beta_pvalue": model.pvalues.get("index_score", np.nan),
        "r_squared":   model.rsquared,
        "n_obs":       int(model.nobs),
        "interpretation": (
            "Green Paradox (extraction acceleration)"
            if model.params.get("index_score", 0) < 0
            else "Fossilflation (investment freeze → supply hoarding)"
        ),
    }

    return results


# =============================================================================
# STAGE 5 — VISUALIZATIONS
# =============================================================================

def plot_weekly_index(weekly: pd.DataFrame, save_path: Path = PLOT_INDEX):
    """Bar chart of the weekly Green Subsidy News Index."""
    fig, ax1 = plt.subplots(figsize=(14, 5))

    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in weekly["index_score"]]
    ax1.bar(weekly["week_start"], weekly["index_score"], color=colors, width=5, alpha=0.8)
    ax1.set_ylabel("Index Score (Σ direction × importance)", fontsize=11)
    ax1.set_xlabel("Date", fontsize=11)
    ax1.set_title("Weekly Green Subsidy News Index (NYT)", fontsize=13, fontweight="bold")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    # Secondary axis: article count
    ax2 = ax1.twinx()
    ax2.plot(weekly["week_start"], weekly["n_articles"], color="#3498db", alpha=0.5, linewidth=1, label="# Articles")
    ax2.set_ylabel("Article Count", fontsize=11, color="#3498db")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Index plot saved: {save_path}")


def plot_regression_scatter(weekly: pd.DataFrame, save_path: Path = PLOT_REGRESSION):
    """Scatter plot: Index Score vs Oil Returns with OLS fit line."""
    if "oil_return" not in weekly.columns:
        log.warning("No oil_return column — skipping regression plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(weekly["index_score"], weekly["oil_return"], alpha=0.5, s=20, c="#2c3e50")

    # OLS fit line
    mask = ~(weekly["index_score"].isna() | weekly["oil_return"].isna())
    if mask.sum() > 2:
        z = np.polyfit(weekly.loc[mask, "index_score"], weekly.loc[mask, "oil_return"], 1)
        p = np.poly1d(z)
        x_range = np.linspace(weekly["index_score"].min(), weekly["index_score"].max(), 100)
        ax.plot(x_range, p(x_range), "r--", linewidth=2, label=f"β = {z[0]:.4f}")
        ax.legend(fontsize=11)

    ax.set_xlabel("Weekly Green Subsidy Index Score", fontsize=11)
    ax.set_ylabel("Weekly Oil Return (ΔP/P)", fontsize=11)
    ax.set_title("Green Subsidy News vs. Oil Price Returns", fontsize=13, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Regression plot saved: {save_path}")


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

async def main():
    log.info("=" * 70)
    log.info("NYT Green Subsidy Index Pipeline — Acharya et al. (2025) Extension")
    log.info("=" * 70)

    # ---- Stage 1: Scrape ----
    log.info("[Stage 1/5] Scraping NYT Archive API...")
    df_raw = await scrape_all_nyt()

    if df_raw.empty:
        log.error("No articles retrieved. Check API key and network. Exiting.")
        return

    df_raw.to_csv(RAW_CSV, index=False)
    log.info(f"Raw articles saved: {RAW_CSV}")

    # ---- Stage 2: LLM Scoring ----
    log.info("[Stage 2/5] Scoring articles via Local AI Stack...")
    df_scored = await score_all_articles(df_raw)
    df_scored.to_csv(RAW_CSV, index=False)  # Overwrite with scores
    log.info(f"Scored articles saved: {RAW_CSV}")

    # ---- Stage 3: Weekly Index ----
    log.info("[Stage 3/5] Building weekly index...")
    weekly = build_weekly_index(df_scored)
    weekly.to_csv(INDEX_CSV, index=False)
    log.info(f"Weekly index saved: {INDEX_CSV}")

    # ---- Stage 4: Regression ----
    log.info("[Stage 4/5] Running high-frequency regression...")
    reg_results = run_regression(weekly)
    pd.DataFrame([reg_results]).to_csv(REGRESSION_CSV, index=False)
    log.info(f"Regression results saved: {REGRESSION_CSV}")

    # ---- Stage 5: Plots ----
    log.info("[Stage 5/5] Generating visualizations...")
    plot_weekly_index(weekly)

    # For regression plot, we need oil_return column (added by run_regression in demo mode)
    weekly_with_oil = weekly.copy()
    if "oil_return" not in weekly_with_oil.columns:
        # Re-run to get the augmented dataframe
        np.random.seed(42)
        n = len(weekly_with_oil)
        beta_synth = -0.005
        weekly_with_oil["oil_return"] = beta_synth * weekly_with_oil["index_score"] + np.random.normal(0, 0.03, n)
    plot_regression_scatter(weekly_with_oil)

    # ---- Summary ----
    log.info("=" * 70)
    log.info("PIPELINE COMPLETE")
    log.info(f"  Articles scraped:  {len(df_raw)}")
    log.info(f"  LLM-relevant:     {(df_scored['relevance'] == 1).sum()}")
    log.info(f"  Index weeks:       {len(weekly)}")
    log.info(f"  β (subsidy→oil):   {reg_results.get('beta', 'N/A'):.6f}")
    log.info(f"  p-value:           {reg_results.get('beta_pvalue', 'N/A'):.4f}")
    log.info(f"  Interpretation:    {reg_results.get('interpretation', 'N/A')}")
    log.info("=" * 70)


if __name__ == "__main__":
    # Windows event loop compatibility
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
