"""
Centralised configuration for the pipeline.

Single Responsibility: all tuneable parameters live here.
Open/Closed: add new settings without modifying consumer modules.
"""

import os
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root (auto-detect)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Compiled keyword patterns (built once, reused everywhere)
# ---------------------------------------------------------------------------
SUBSIDY_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bsubsid(?:y|ies|ize|ized|izing)\b", re.IGNORECASE),
    re.compile(r"\btax\s+credit(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bgrant(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bincentive(?:s)?\b", re.IGNORECASE),
    re.compile(r"\brebate(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bloan\s+guarantee\b", re.IGNORECASE),
    re.compile(r"\bproduction\s+tax\s+credit\b", re.IGNORECASE),
    re.compile(r"\binvestment\s+tax\s+credit\b", re.IGNORECASE),
    re.compile(r"\bfeed[- ]in[- ]tariff\b", re.IGNORECASE),
    re.compile(r"\bgreen\s+industrial\s+policy\b", re.IGNORECASE),
    re.compile(r"\bclimate\s+(?:bill|package|act|law|legislation)\b", re.IGNORECASE),
]

ENERGY_PATTERNS: list[re.Pattern] = [
    re.compile(r"\brenewable(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bsolar\b", re.IGNORECASE),
    re.compile(r"\bwind\s+(?:energy|power|farm|turbine|project)s?\b", re.IGNORECASE),
    re.compile(r"\bwind\b", re.IGNORECASE),
    re.compile(r"\bgreen\s+energy\b", re.IGNORECASE),
    re.compile(r"\bclean\s+energy\b", re.IGNORECASE),
    re.compile(r"\bclean\s+power\b", re.IGNORECASE),
    re.compile(r"\binflation\s+reduction\s+act\b", re.IGNORECASE),
    re.compile(r"\bIRA\b"),
    re.compile(r"\bEUA(?:s)?\b"),
    re.compile(r"\benergy\s+transition\b", re.IGNORECASE),
    re.compile(r"\belectric\s+vehicl\w*\b", re.IGNORECASE),
    re.compile(r"\bbatter(?:y|ies)\b", re.IGNORECASE),
    re.compile(r"\bhydrogen\b", re.IGNORECASE),
    re.compile(r"\boffshore\s+wind\b", re.IGNORECASE),
    re.compile(r"\bcarbon\b", re.IGNORECASE),
    re.compile(r"\bemission(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bfossil\s+fuel\b", re.IGNORECASE),
    re.compile(r"\boil\b", re.IGNORECASE),
    re.compile(r"\bnatural\s+gas\b", re.IGNORECASE),
    re.compile(r"\bnuclear\b", re.IGNORECASE),
    re.compile(r"\bheat\s+pump\b", re.IGNORECASE),
    re.compile(r"\benergy\s+efficien\w*\b", re.IGNORECASE),
]

ALLOWED_SECTIONS: set[str] = {
    "business", "climate", "energy", "national", "financial",
    "us", "world", "science", "your money",
    "politics", "washington", "environment",
}


# ---------------------------------------------------------------------------
# Pipeline settings (immutable dataclass)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineConfig:
    # Local AI Stack
    local_ai_base_url: str = "http://127.0.0.1:9485/v1"
    local_model_name: str = "local-model"

    # NYT API — loaded from .env automatically
    nyt_api_key: str = field(default_factory=lambda: os.getenv("NYT_API_KEY", "").strip())
    nyt_api_secret: str = field(default_factory=lambda: os.getenv("NYT_API_SECRET", "").strip())
    nyt_rate_limit_delay: float = 12.0  # seconds between Archive API calls

    # FRED API — loaded from .env
    fred_api_key: str = field(default_factory=lambda: os.getenv("FRED_API_KEY", "").strip())

    # Hardware throttle (AMD 8 GB shared VRAM)
    max_concurrent_llm: int = 3

    # Scraping scope
    scrape_years: list[int] = field(default_factory=lambda: [2022, 2023, 2024])

    # Output paths
    output_dir: Path = Path(".")

    @property
    def raw_csv(self) -> Path:
        return self.output_dir / "nyt_articles_raw.csv"

    @property
    def index_csv(self) -> Path:
        return self.output_dir / "weekly_green_subsidy_index.csv"

    @property
    def regression_csv(self) -> Path:
        return self.output_dir / "regression_results.csv"

    @property
    def has_nyt_key(self) -> bool:
        return bool(self.nyt_api_key) and self.nyt_api_key != "VOTRE_CLE_API_NYT_ICI"

    @property
    def has_fred_key(self) -> bool:
        return bool(self.fred_api_key)
