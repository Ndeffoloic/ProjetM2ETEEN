"""
LLM-based article scorer via Local AI Stack.

Single Responsibility: scores articles for relevance, direction, importance.
Interface Segregation: exposes only `score_articles(df, cfg)`.
"""

import asyncio
import json
import re

import pandas as pd
from openai import AsyncOpenAI

from .config import PipelineConfig, log

# Try optional tqdm
try:
    from tqdm.asyncio import tqdm as async_tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

LLM_SYSTEM_PROMPT = """You are a financial climate-risk analyst specializing in energy transition policy.
You will receive a news snippet. Evaluate it and respond with ONLY a JSON object:

{
  "relevance": <0 or 1>,
  "direction": <-1, 0, or 1>,
  "importance": <1, 2, or 3>,
  "rationale": "<one sentence>"
}

Definitions:
- relevance: 1 if the article discusses government subsidies, tax credits, or grants for renewable/clean energy at a national or supranational level. 0 otherwise.
- direction: 1 if subsidies are being introduced, increased, or extended. -1 if cut, repealed, or blocked. 0 if neutral/ambiguous.
- importance: 1 = minor (local grants). 2 = major (multi-billion federal). 3 = massive/systemic (IRA, EU Green Deal).
- rationale: one sentence explaining your decision.

Output ONLY the JSON. No markdown, no commentary."""


def _build_prompt(article: dict) -> str:
    return (
        f"Headline: {article['headline']}\n"
        f"Snippet: {article['snippet']}\n"
        f"Lead: {article.get('lead_paragraph', '')}\n"
        f"Date: {article['date']}"
    )


async def _score_one(
    article: dict,
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
    retries: int = 2,
) -> dict:
    """Score a single article (semaphore-gated for VRAM safety)."""
    async with semaphore:
        prompt = _build_prompt(article)

        for attempt in range(retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": LLM_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=256,
                )
                raw = response.choices[0].message.content.strip()

                # Handle markdown code blocks
                if raw.startswith("```"):
                    raw = re.sub(r"```(?:json)?\s*", "", raw).strip("`").strip()

                parsed = json.loads(raw)

                article["relevance"] = int(parsed.get("relevance", 0)) if parsed.get("relevance") in (0, 1, "0", "1") else 0
                article["direction"] = max(-1, min(1, int(parsed.get("direction", 0))))
                article["importance"] = max(1, min(3, int(parsed.get("importance", 1))))
                article["rationale"] = str(parsed.get("rationale", ""))
                return article

            except json.JSONDecodeError as e:
                log.warning(f"JSON parse error (attempt {attempt + 1}): {e}")
                if attempt == retries:
                    article.update({"relevance": 0, "direction": 0, "importance": 0, "rationale": f"JSON_ERROR: {raw[:100]}"})
            except Exception as e:
                log.warning(f"LLM error (attempt {attempt + 1}): {e}")
                if attempt == retries:
                    article.update({"relevance": 0, "direction": 0, "importance": 0, "rationale": f"LLM_ERROR: {str(e)[:100]}"})
                else:
                    await asyncio.sleep(2 ** attempt)

    return article


async def score_articles(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Score all articles with hardware-gated concurrency."""
    client = AsyncOpenAI(base_url=cfg.local_ai_base_url, api_key="not-needed-for-local")
    semaphore = asyncio.Semaphore(cfg.max_concurrent_llm)
    articles = df.to_dict("records")

    log.info(f"Scoring {len(articles)} articles (max {cfg.max_concurrent_llm} concurrent)...")

    tasks = [_score_one(art, client, cfg.local_model_name, semaphore) for art in articles]

    if HAS_TQDM:
        scored = await async_tqdm.gather(*tasks, desc="LLM Scoring")
    else:
        scored = await asyncio.gather(*tasks)

    result = pd.DataFrame(scored)
    n_relevant = (result["relevance"] == 1).sum()
    log.info(f"Scoring complete: {n_relevant}/{len(result)} marked relevant")
    return result
