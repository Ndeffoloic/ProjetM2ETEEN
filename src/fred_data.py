"""
FRED API data fetcher for energy prices.

Single Responsibility: fetches and aligns electricity and oil price series.
Dependency Inversion: depends on PipelineConfig for API key.

Series used:
  - APU000072610 : Average electricity price ($/kWh), monthly, US city average
  - WCOILWTICO   : WTI crude oil price ($/barrel), weekly
  - DHHNGSP      : Henry Hub natural gas spot price ($/MMBtu), weekly
                   ★ Gas is the marginal fuel for ~40% of US electricity,
                     making elec/gas a far more theoretically grounded
                     substitution ratio than elec/oil (oil = 0.5% of US elec mix).

The module computes:
  - ratio_elec_oil = (elec_price * 1000) / oil_price (legacy)
  - ratio_elec_gas = elec_price / gas_price * 100  (theory-aligned)
  - Weekly returns (pct change) for regression
"""

import json
import time
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

from .config import PipelineConfig, log


# ---------------------------------------------------------------------------
# FRED API helpers
# ---------------------------------------------------------------------------

_FRED_TIMEOUT_SEC = 60
_FRED_MAX_RETRIES = 3
_FRED_BACKOFF_SEC = 5  # exponential backoff base


def _fetch_fred_series(
    series_id: str,
    api_key: str,
    start: str = "2012-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch a single FRED series as a DataFrame with columns [date, value]."""
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}"
        f"&api_key={api_key}"
        f"&file_type=json"
        f"&observation_start={start}"
        f"&observation_end={end}"
    )

    log.info(f"Fetching FRED series: {series_id} ({start} -> {end})")

    # Retry with exponential backoff to survive transient WinError 10060 / 5xx
    data = None
    last_err: Exception | None = None
    for attempt in range(1, _FRED_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NYT-Pipeline/1.0"})
            with urllib.request.urlopen(req, timeout=_FRED_TIMEOUT_SEC) as resp:
                data = json.loads(resp.read())
            break  # success
        except Exception as e:
            last_err = e
            if attempt < _FRED_MAX_RETRIES:
                wait = _FRED_BACKOFF_SEC * (2 ** (attempt - 1))
                log.warning(
                    f"FRED {series_id} attempt {attempt}/{_FRED_MAX_RETRIES} failed: {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                log.error(f"FRED API error for {series_id} after {attempt} attempts: {e}")
                return pd.DataFrame(columns=["date", "value"])

    if data is None:
        log.error(f"FRED API unreachable for {series_id}: {last_err}")
        return pd.DataFrame(columns=["date", "value"])

    observations = data.get("observations", [])
    if not observations:
        log.warning(f"No observations returned for {series_id}")
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(observations)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    log.info(f"  {series_id}: {len(df)} observations ({df['date'].min().date()} -> {df['date'].max().date()})")
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_energy_prices(
    cfg: PipelineConfig,
    start: str = "2012-01-01",
) -> pd.DataFrame:
    """
    Fetch electricity and oil prices from FRED, compute weekly ratio.

    Returns DataFrame with columns:
        week_start,
        elec_price, oil_price, gas_price,
        ratio_elec_oil, ratio_elec_gas,
        elec_return, oil_return, gas_return,
        ratio_return, ratio_gas_return
    """
    api_key = cfg.fred_api_key
    if not api_key:
        log.error("FRED_API_KEY not set in .env")
        return pd.DataFrame()

    # --- Fetch raw series ---
    df_elec = _fetch_fred_series("APU000072610", api_key, start=start)
    df_oil_w = _fetch_fred_series("WCOILWTICO", api_key, start=start)
    # Henry Hub natural gas spot — weekly. THIS is the marginal electricity fuel.
    df_gas_w = _fetch_fred_series("DHHNGSP", api_key, start=start)

    if df_elec.empty or df_oil_w.empty:
        log.error("Failed to fetch one or both core FRED series (elec/oil).")
        return pd.DataFrame()

    df_elec = df_elec.rename(columns={"value": "elec_price"})
    df_oil_w = df_oil_w.rename(columns={"value": "oil_price"})

    has_gas = not df_gas_w.empty
    if has_gas:
        df_gas_w = df_gas_w.rename(columns={"value": "gas_price"})
    else:
        log.warning("Natural gas series unavailable — ratio_elec_gas will be skipped.")

    # --- Electricity is monthly: forward-fill to weekly ---
    df_elec = df_elec.set_index("date").resample("D").ffill().reset_index()

    # W-SUN = weeks ending Sunday → start_time = Monday (aligns with index_builder's W-MON freq)
    df_elec["week_start"] = df_elec["date"].dt.to_period("W-SUN").apply(lambda p: p.start_time)
    df_oil_w["week_start"] = df_oil_w["date"].dt.to_period("W-SUN").apply(lambda p: p.start_time)
    if has_gas:
        df_gas_w["week_start"] = df_gas_w["date"].dt.to_period("W-SUN").apply(lambda p: p.start_time)

    # Aggregate to weekly (mean within each week)
    elec_weekly = df_elec.groupby("week_start")["elec_price"].mean().reset_index()
    oil_weekly = df_oil_w.groupby("week_start")["oil_price"].mean().reset_index()

    # --- Merge ---
    merged = elec_weekly.merge(oil_weekly, on="week_start", how="inner")
    if has_gas:
        gas_weekly = df_gas_w.groupby("week_start")["gas_price"].mean().reset_index()
        # left join: keep all elec/oil weeks even if gas missing on some
        merged = merged.merge(gas_weekly, on="week_start", how="left")

    merged = merged.sort_values("week_start").reset_index(drop=True)

    if merged.empty:
        log.error("No overlapping data between FRED series.")
        return pd.DataFrame()

    # --- Compute ratios ---
    # Elec ($/kWh ~0.10-0.20) / Oil ($/bbl ~50-120) — scaled ×1000 for readability
    merged["ratio_elec_oil"] = (merged["elec_price"] * 1000) / merged["oil_price"]

    # Elec ($/kWh) / Gas ($/MMBtu ~2-8) — scaled ×100. Far more theory-aligned.
    if has_gas:
        merged["ratio_elec_gas"] = (merged["elec_price"] * 100) / merged["gas_price"]

    # --- Weekly returns (percentage change) ---
    merged["elec_return"] = merged["elec_price"].pct_change()
    merged["oil_return"] = merged["oil_price"].pct_change()
    merged["ratio_return"] = merged["ratio_elec_oil"].pct_change()  # legacy name = elec/oil

    if has_gas:
        merged["gas_return"] = merged["gas_price"].pct_change()
        merged["ratio_gas_return"] = merged["ratio_elec_gas"].pct_change()

    # Drop rows missing the core columns; preserve gas where possible
    core = ["index"] if "index" in merged.columns else []
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=["elec_return", "oil_return", "ratio_return"]).reset_index(drop=True)

    log.info(
        f"Energy prices aligned: {len(merged)} weeks "
        f"({merged['week_start'].min().date()} -> {merged['week_start'].max().date()})"
    )
    msg = (
        f"  Electricity: {merged['elec_price'].mean():.4f} $/kWh avg | "
        f"Oil: ${merged['oil_price'].mean():.1f}/bbl avg | "
        f"Ratio E/O: {merged['ratio_elec_oil'].mean():.2f} avg"
    )
    if has_gas and "gas_price" in merged.columns:
        msg += (
            f" | Gas: ${merged['gas_price'].mean():.2f}/MMBtu avg | "
            f"Ratio E/G: {merged['ratio_elec_gas'].mean():.2f} avg"
        )
    log.info(msg)

    return merged
