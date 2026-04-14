"""
Article filtering logic.

Single Responsibility: only decides whether an article is relevant.
Dependency Inversion: depends on pattern lists from config, not concrete regexes.
"""

from .config import SUBSIDY_PATTERNS, ENERGY_PATTERNS, ALLOWED_SECTIONS


def passes_keyword_filter(text: str) -> bool:
    """Require at least one subsidy-related AND one energy-related keyword."""
    has_subsidy = any(p.search(text) for p in SUBSIDY_PATTERNS)
    has_energy = any(p.search(text) for p in ENERGY_PATTERNS)
    return has_subsidy and has_energy


def passes_section_filter(article: dict) -> bool:
    """Check if the article belongs to an allowed NYT section."""
    section = article.get("section_name", "").lower().strip()
    news_desk = article.get("news_desk", "").lower().strip()
    return section in ALLOWED_SECTIONS or news_desk in ALLOWED_SECTIONS
