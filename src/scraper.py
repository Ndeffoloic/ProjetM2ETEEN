"""
Async NYT Archive API scraper.

Single Responsibility: fetches and pre-filters NYT articles.
Open/Closed: date range and filters are injected via PipelineConfig.
"""

import asyncio
import aiohttp
import pandas as pd

from .config import PipelineConfig, log
from .filters import passes_section_filter, passes_keyword_filter


async def _fetch_month(
    session: aiohttp.ClientSession,
    year: int,
    month: int,
    cfg: PipelineConfig,
    rate_limiter: asyncio.Semaphore,
) -> list[dict]:
    """Fetch one month of NYT Archive data (rate-limited)."""
    async with rate_limiter:
        url = (
            f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"
            f"?api-key={cfg.nyt_api_key}"
        )
        log.info(f"Fetching NYT archive: {year}-{month:02d}")

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 429:
                    log.warning(f"Rate-limited on {year}-{month:02d}, retrying after 60 s...")
                    await asyncio.sleep(60)
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as retry:
                        if retry.status != 200:
                            log.error(f"Retry failed {year}-{month:02d}: HTTP {retry.status}")
                            return []
                        data = await retry.json()
                elif resp.status != 200:
                    log.error(f"HTTP {resp.status} for {year}-{month:02d}")
                    return []
                else:
                    data = await resp.json()
        except Exception as e:
            log.error(f"Network error {year}-{month:02d}: {e}")
            return []

        articles = data.get("response", {}).get("docs", [])
        filtered = []

        for art in articles:
            if not passes_section_filter(art):
                continue

            headline = art.get("headline", {}).get("main", "")
            snippet = art.get("snippet", "") or ""
            lead = art.get("lead_paragraph", "") or ""
            full_text = f"{headline} {snippet} {lead}"

            if not passes_keyword_filter(full_text):
                continue

            filtered.append({
                "date": art.get("pub_date", ""),
                "headline": headline,
                "snippet": snippet,
                "lead_paragraph": lead,
                "section": art.get("section_name", ""),
                "news_desk": art.get("news_desk", ""),
                "web_url": art.get("web_url", ""),
                "text_to_analyze": full_text,
            })

        log.info(f"  {year}-{month:02d}: {len(filtered)} relevant / {len(articles)} total")
        await asyncio.sleep(cfg.nyt_rate_limit_delay)
        return filtered


async def scrape_nyt(cfg: PipelineConfig) -> pd.DataFrame:
    """Orchestrate full NYT scraping with rate limiting."""
    rate_limiter = asyncio.Semaphore(1)  # sequential for API safety
    all_articles: list[dict] = []

    async with aiohttp.ClientSession() as session:
        tasks = [
            _fetch_month(session, year, month, cfg, rate_limiter)
            for year in cfg.scrape_years
            for month in range(1, 13)
        ]
        results = await asyncio.gather(*tasks)

    for batch in results:
        all_articles.extend(batch)

    df = pd.DataFrame(all_articles)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    log.info(f"Total articles after filtering: {len(df)}")
    return df
