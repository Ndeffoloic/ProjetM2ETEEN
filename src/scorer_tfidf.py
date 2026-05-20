"""
TF-IDF + Logistic Regression scorer (CPU-only, zero GPU).

Single Responsibility: scores articles for relevance, direction, importance
using classical ML instead of a local LLM.

Strategy:
  1. A small bootstrapped training set of ~120 hand-labeled examples is embedded
     in this module (representative headlines from 2022-2024 NYT climate coverage).
  2. TF-IDF vectorizer + Logistic Regression trained on first call, then cached.
  3. Each article gets: relevance (0/1), direction (-1/0/1), importance (1/2/3).
  4. Hybrid: symbolic rules correct high-confidence edge cases.

Open/Closed: swap this scorer for the LLM scorer without touching the pipeline.
"""

import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import log

# ---------------------------------------------------------------------------
# Bootstrapped training data (curated from real NYT headline patterns)
# Each entry: (text, relevance, direction)
# ---------------------------------------------------------------------------
_TRAINING_DATA: list[tuple[str, int, int]] = [
    # =======================================================================
    # RELEVANT, DIRECTION = +1 (subsidy increase / new program)
    # Covers: Obama 2013-2016, Biden 2021-2024, EU, international, 2025-2026
    # =======================================================================

    # --- Obama era (2013-2016): Clean Power Plan, extensions, DOE grants ---
    ("Obama administration proposes Clean Power Plan with incentives for renewables", 1, 1),
    ("Congress extends production tax credit for wind energy in budget deal", 1, 1),
    ("Department of Energy awards $4 billion in loan guarantees for solar projects", 1, 1),
    ("White House announces new initiatives to boost solar energy in low-income areas", 1, 1),
    ("Bipartisan agreement extends solar investment tax credit through 2021", 1, 1),
    ("Obama signs omnibus bill with five-year extension of renewable energy tax credits", 1, 1),
    ("Federal grants expand community solar programs across 12 states", 1, 1),
    ("US joins Paris Agreement pledging billions for clean energy development", 1, 1),
    ("Administration increases funding for ARPA-E advanced energy research", 1, 1),
    ("New DOE program offers subsidized loans for offshore wind development", 1, 1),

    # --- Biden era (2021-2024): IRA, infrastructure bill, massive expansion ---
    ("Biden signs Inflation Reduction Act with $369 billion in clean energy subsidies", 1, 1),
    ("Senate approves massive expansion of solar tax credits", 1, 1),
    ("New federal grants to boost wind energy production across rural America", 1, 1),
    ("Administration extends production tax credits for renewable energy through 2032", 1, 1),
    ("Congress passes historic clean energy investment package", 1, 1),
    ("Tax credits for electric vehicles and solar panels expanded significantly", 1, 1),
    ("Government launches $50 billion green hydrogen subsidy program", 1, 1),
    ("New legislation doubles federal funding for renewable energy research", 1, 1),
    ("States compete for billions in new clean energy manufacturing subsidies", 1, 1),
    ("IRA tax credits accelerate solar deployment beyond expectations", 1, 1),
    ("Federal investment tax credit extended for offshore wind projects", 1, 1),
    ("Treasury releases guidance expanding clean energy tax credit eligibility", 1, 1),
    ("Biden administration announces $7 billion solar subsidy for low-income communities", 1, 1),
    ("Production tax credit for wind energy renewed for another decade", 1, 1),
    ("New grants fund battery storage alongside solar installations nationwide", 1, 1),
    ("Climate bill includes unprecedented subsidies for green industrial policy", 1, 1),
    ("Federal rebates for heat pumps and energy efficiency now available", 1, 1),
    ("Major clean energy subsidies included in bipartisan infrastructure bill", 1, 1),
    ("Government subsidy makes solar power cheaper than coal in key markets", 1, 1),
    ("New incentive program pays farmers to install wind turbines", 1, 1),
    ("Tax credit boost for domestic solar panel manufacturing under IRA", 1, 1),
    ("Record federal spending on renewable energy grants in fiscal year 2023", 1, 1),

    # --- EU & International (2013-2026) ---
    ("EU Green Deal allocates 250 billion euros for clean energy transition", 1, 1),
    ("European Commission approves new renewable energy subsidy framework", 1, 1),
    ("Japan triples green energy subsidies in latest economic stimulus", 1, 1),
    ("Germany passes landmark renewable energy law with guaranteed feed-in tariffs", 1, 1),
    ("UK announces contracts for difference boosting offshore wind subsidies", 1, 1),
    ("India launches $26 billion green energy incentive program", 1, 1),
    ("China increases renewable energy subsidies in latest five-year plan", 1, 1),
    ("EU REPowerEU plan allocates additional funds for solar and wind", 1, 1),
    ("South Korea commits $35 billion to green new deal energy subsidies", 1, 1),
    ("Canada introduces clean electricity tax credits mirroring US IRA", 1, 1),

    # --- 2025-2026: anticipated events ---
    ("New administration preserves core IRA clean energy tax credits despite review", 1, 1),
    ("Bipartisan group secures extension of advanced manufacturing clean energy credits", 1, 1),
    ("EU expands Net-Zero Industry Act subsidies for domestic solar manufacturing", 1, 1),
    ("Congressional deal includes $20 billion for next-generation nuclear subsidies", 1, 1),
    ("Global climate summit yields new multilateral green subsidy commitments", 1, 1),

    # =======================================================================
    # RELEVANT, DIRECTION = -1 (subsidy cut / repeal / rollback)
    # Covers: Trump 2017-2020, GOP opposition, post-2024 uncertainty
    # =======================================================================

    # --- Trump era (2017-2020): rollbacks, executive orders, deregulation ---
    ("Trump signs executive order rolling back Obama-era clean energy regulations", 1, -1),
    ("Administration proposes eliminating renewable energy tax credits in budget", 1, -1),
    ("Trump revokes Clean Power Plan gutting incentives for wind and solar", 1, -1),
    ("White House budget slashes Department of Energy clean energy programs by 72 percent", 1, -1),
    ("Administration moves to end electric vehicle tax credit in fiscal proposal", 1, -1),
    ("Trump pulls United States out of Paris climate accord signaling subsidy retreat", 1, -1),
    ("Interior Department opens Arctic refuge to drilling reversing conservation subsidies", 1, -1),
    ("EPA rolls back methane regulations reducing compliance incentives for renewables", 1, -1),
    ("Administration approves Keystone XL pipeline while cutting clean energy grants", 1, -1),
    ("Trump imposes tariffs on imported solar panels undermining subsidy economics", 1, -1),
    ("Federal clean energy loan program frozen under new administration directives", 1, -1),
    ("ARPA-E budget zeroed out in presidential budget request for third year", 1, -1),
    ("Administration withdraws from Green Climate Fund halting international clean energy aid", 1, -1),
    ("Commerce Department tariffs on Chinese solar cells raise costs despite tax credits", 1, -1),
    ("Wind energy production tax credit allowed to expire without congressional renewal", 1, -1),

    # --- GOP opposition / general cuts (2013-2026) ---
    ("Republicans push to repeal clean energy tax credits in new bill", 1, -1),
    ("Proposed budget cuts threaten renewable energy subsidies", 1, -1),
    ("GOP candidates vow to end wind energy subsidies if elected", 1, -1),
    ("Solar subsidy program faces elimination in spending negotiations", 1, -1),
    ("Federal wind energy grants expire with no renewal in sight", 1, -1),
    ("Court ruling blocks implementation of key IRA clean energy provisions", 1, -1),
    ("State legislature votes to end renewable portfolio standard and subsidies", 1, -1),
    ("Industry warns of layoffs as green energy tax credits face phase-out", 1, -1),
    ("Administration scales back scope of electric vehicle tax credits", 1, -1),
    ("Bipartisan opposition threatens to kill expanded solar subsidies", 1, -1),
    ("European austerity measures target renewable energy support schemes", 1, -1),
    ("Congressional review finds waste in clean energy grant programs", 1, -1),
    ("New restrictions limit eligibility for clean energy manufacturing credits", 1, -1),
    ("Political backlash against wind farms leads to subsidy freezes in key states", 1, -1),

    # --- 2025-2026 uncertainty ---
    ("New administration orders review of all IRA clean energy provisions", 1, -1),
    ("Trump vows to end all green energy subsidies on day one", 1, -1),
    ("Executive order pauses disbursement of remaining IRA clean energy funds", 1, -1),
    ("Proposed legislation would claw back unspent Inflation Reduction Act subsidies", 1, -1),
    ("States sue to block federal rollback of renewable energy incentive programs", 1, -1),
    ("Offshore wind projects cancelled as developers lose confidence in subsidy stability", 1, -1),

    # =======================================================================
    # RELEVANT, DIRECTION = 0 (neutral / ambiguous / debate)
    # =======================================================================
    ("Debate intensifies over future of renewable energy subsidies in Congress", 1, 0),
    ("Economists disagree on effectiveness of green energy tax credits", 1, 0),
    ("Renewable subsidy reform under discussion but no decision reached", 1, 0),
    ("Mixed signals from Washington on clean energy funding priorities", 1, 0),
    ("Energy committee holds hearing on subsidy design for next decade", 1, 0),
    ("Analysis shows uneven distribution of clean energy subsidies across states", 1, 0),
    ("Both parties claim credit for renewable energy investment growth", 1, 0),
    ("Think tank proposes restructuring rather than expanding green subsidies", 1, 0),
    ("Uncertainty clouds outlook for clean energy tax credit extensions", 1, 0),
    ("Industry divided on preferred design for next generation of energy subsidies", 1, 0),
    ("CBO report questions long-term fiscal cost of unlimited clean energy tax credits", 1, 0),
    ("Lobbying battle erupts over which technologies qualify for green subsidies", 1, 0),
    ("Transition team sends conflicting signals on clean energy subsidy future", 1, 0),
    ("Red states benefit most from IRA subsidies complicating Republican repeal efforts", 1, 0),
    ("Senate holds confirmation hearing debating nominees stance on energy subsidies", 1, 0),

    # =======================================================================
    # NOT RELEVANT (energy/climate but NOT about subsidy policy changes)
    # =======================================================================

    # --- Oil & gas market events ---
    ("Oil prices surge as OPEC announces production cuts", 0, 0),
    ("Natural gas prices fall on mild winter forecast", 0, 0),
    ("Pipeline explosion disrupts oil supply in Gulf of Mexico", 0, 0),
    ("Gasoline prices at the pump hit summer highs", 0, 0),
    ("Major oil discovery in Guyana reshapes regional energy dynamics", 0, 0),
    ("Russia gas cutoff forces Europe to accelerate energy transition", 0, 0),
    ("Saudi Arabia extends voluntary oil production cuts through Q1", 0, 0),
    ("US crude oil exports reach record high amid global demand surge", 0, 0),
    ("Brent crude falls below $70 on recession fears", 0, 0),

    # --- Climate science & weather ---
    ("Wildfires devastate California as drought worsens", 0, 0),
    ("Scientists warn of accelerating Arctic ice melt", 0, 0),
    ("Hurricane season forecast predicts above-average activity", 0, 0),
    ("IPCC report calls for drastic emissions reductions by 2030", 0, 0),
    ("Record heatwave strains power grids across southern Europe", 0, 0),
    ("Drought threatens hydropower generation in western states", 0, 0),
    ("Global carbon emissions reach new record high in 2023", 0, 0),
    ("UN climate talks stall over loss and damage funding dispute", 0, 0),

    # --- Corporate / technology ---
    ("Tesla reports record quarterly deliveries worldwide", 0, 0),
    ("ExxonMobil reports quarterly earnings beating expectations", 0, 0),
    ("Solar panel efficiency record broken by MIT researchers", 0, 0),
    ("China dominates global solar panel manufacturing supply chain", 0, 0),
    ("Electric vehicle sales surpass gas cars in Norway", 0, 0),
    ("Nuclear power debate resurfaces as climate tool", 0, 0),
    ("Carbon capture project in Texas begins commercial operations", 0, 0),
    ("Energy storage breakthrough could transform renewable reliability", 0, 0),
    ("Inflation drives up costs for wind turbine components", 0, 0),
    ("Wall Street banks increase fossil fuel financing despite pledges", 0, 0),
    ("Lithium mining boom raises environmental concerns in South America", 0, 0),
    ("Global investment in clean energy tops $500 billion for first time", 0, 0),

    # --- Regulation (non-subsidy) ---
    ("EU carbon border adjustment mechanism takes effect", 0, 0),
    ("New study links fracking to local water contamination", 0, 0),
    ("Coal plant closures accelerate across the Midwest", 0, 0),
    ("Utility companies invest in grid modernization projects", 0, 0),
    ("Activists block construction of new gas pipeline in Appalachia", 0, 0),
    ("EPA finalizes new power plant emissions rules without subsidy provisions", 0, 0),
    ("California bans sale of new gasoline cars after 2035", 0, 0),
    ("SEC finalizes climate disclosure rules for public companies", 0, 0),
    ("Supreme Court limits EPA authority to regulate power plant emissions", 0, 0),
    ("Federal Reserve warns of financial risks from stranded fossil fuel assets", 0, 0),
]


# ---------------------------------------------------------------------------
# Symbolic rules for high-confidence override
# ---------------------------------------------------------------------------
_POS_RULES = [
    (re.compile(r"\b(sign|pass|approv|extend|expand|boost|launch|increas|doubl|tripl|renew|fund|award|secur)\w*\b", re.I),
     re.compile(r"\b(subsid|tax\s+credit|grant|incentiv|rebat|IRA|green\s+deal|clean\s+energy|feed[- ]in|loan\s+guarantee)\w*\b", re.I)),
]

_NEG_RULES = [
    (re.compile(r"\b(repeal|cut|block|eliminat|end|halt|cancel|scrap|phase\s+out|oppos|slash|roll\s*back|revok|withdraw|freez|suspend|pause|claw\s*back|gut|zero\s+out|kill)\w*\b", re.I),
     re.compile(r"\b(subsid|tax\s+credit|grant|incentiv|rebat|clean\s+energy|renewable|green|solar|wind|IRA|clean\s+power)\w*\b", re.I)),
]

_IMPORTANCE_HIGH = re.compile(
    r"\b(inflation\s+reduction\s+act|IRA|green\s+deal|clean\s+power\s+plan|paris\s+(agreement|accord)"
    r"|historic|landmark|trillion|unprecedented|net[- ]zero\s+industry|REPowerEU)\b", re.I
)
_IMPORTANCE_MED = re.compile(
    r"\b(billion|federal|national|major|massive|significant|executive\s+order|bipartisan|omnibus)\b", re.I
)


def _apply_symbolic_rules(text: str) -> tuple[int | None, int | None]:
    """Return (direction_override, importance) or (None, None) if no rule fires."""
    direction = None
    for action_re, target_re in _POS_RULES:
        if action_re.search(text) and target_re.search(text):
            direction = 1
            break
    if direction is None:
        for action_re, target_re in _NEG_RULES:
            if action_re.search(text) and target_re.search(text):
                direction = -1
                break

    importance = None
    if _IMPORTANCE_HIGH.search(text):
        importance = 3
    elif _IMPORTANCE_MED.search(text):
        importance = 2

    return direction, importance


# ---------------------------------------------------------------------------
# Model training (lazy singleton)
# ---------------------------------------------------------------------------
_RELEVANCE_MODEL: Pipeline | None = None
_DIRECTION_MODEL: Pipeline | None = None
_MODEL_CACHE = Path("models_cache")


def _prepare_training_data() -> tuple[list[str], np.ndarray, np.ndarray]:
    texts = [t[0] for t in _TRAINING_DATA]
    relevance = np.array([t[1] for t in _TRAINING_DATA])
    direction = np.array([t[2] for t in _TRAINING_DATA])
    return texts, relevance, direction


def _build_tfidf_pipeline(n_classes: int = 2) -> Pipeline:
    """Build a TF-IDF + LR pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=20_000,
            stop_words="english",
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])


def _get_models() -> tuple[Pipeline, Pipeline]:
    """Train or load cached models."""
    global _RELEVANCE_MODEL, _DIRECTION_MODEL

    if _RELEVANCE_MODEL is not None and _DIRECTION_MODEL is not None:
        return _RELEVANCE_MODEL, _DIRECTION_MODEL

    # Try cache
    cache_path = _MODEL_CACHE / "tfidf_models.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                _RELEVANCE_MODEL, _DIRECTION_MODEL = pickle.load(f)
            log.info("Loaded cached TF-IDF models.")
            return _RELEVANCE_MODEL, _DIRECTION_MODEL
        except Exception:
            log.warning("Cache corrupted, retraining...")

    texts, relevance, direction = _prepare_training_data()

    # Model 1: relevance (binary)
    log.info("Training relevance model (TF-IDF + LR)...")
    _RELEVANCE_MODEL = _build_tfidf_pipeline(n_classes=2)
    _RELEVANCE_MODEL.fit(texts, relevance)

    # Model 2: direction (multiclass -1, 0, 1) — only on relevant articles
    rel_mask = relevance == 1
    rel_texts = [t for t, r in zip(texts, relevance) if r == 1]
    rel_dirs = direction[rel_mask]

    log.info("Training direction model (TF-IDF + LR, 3-class)...")
    _DIRECTION_MODEL = _build_tfidf_pipeline(n_classes=3)
    _DIRECTION_MODEL.fit(rel_texts, rel_dirs)

    # Cache
    _MODEL_CACHE.mkdir(exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((_RELEVANCE_MODEL, _DIRECTION_MODEL), f)
    log.info("Models trained and cached.")

    return _RELEVANCE_MODEL, _DIRECTION_MODEL


# ---------------------------------------------------------------------------
# Public scoring function (same interface as scorer.py)
# ---------------------------------------------------------------------------
def score_articles_tfidf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score articles using TF-IDF + LR + symbolic rules.
    Adds columns: relevance, direction, importance, rationale.
    """
    if df.empty:
        return df

    rel_model, dir_model = _get_models()

    texts = df["text_to_analyze"].fillna("").tolist()

    # --- Relevance ---
    rel_proba = rel_model.predict_proba(texts)
    rel_pred = rel_model.predict(texts)
    # Probability of being relevant
    rel_confidence = rel_proba[:, 1] if rel_proba.shape[1] == 2 else rel_proba.max(axis=1)

    # --- Direction (for all, then mask) ---
    dir_pred = dir_model.predict(texts)
    dir_proba = dir_model.predict_proba(texts)
    # Get max class probability as confidence
    dir_confidence = dir_proba.max(axis=1)

    # --- Assemble results ---
    results = df.copy()
    results["relevance"] = rel_pred.astype(int)
    results["direction"] = dir_pred.astype(int)
    results["rel_confidence"] = rel_confidence
    results["dir_confidence"] = dir_confidence

    # --- Symbolic rule overrides for high-confidence cases ---
    importances = []
    rationales = []

    for idx, row in results.iterrows():
        text = row["text_to_analyze"]
        sym_dir, sym_imp = _apply_symbolic_rules(text)

        # Override direction if rule fires AND model is uncertain
        if sym_dir is not None and row["dir_confidence"] < 0.7:
            results.at[idx, "direction"] = sym_dir
            rationale = f"rule_override(dir={sym_dir})"
        elif sym_dir is not None and row["direction"] == sym_dir:
            rationale = f"model+rule_agree(dir={sym_dir})"
        else:
            rationale = f"model(rel={row['rel_confidence']:.2f},dir_conf={row['dir_confidence']:.2f})"

        # Importance from rules, fallback to heuristic
        if sym_imp is not None:
            importances.append(sym_imp)
        else:
            importances.append(1)

        rationales.append(rationale)

    results["importance"] = importances
    results["rationale"] = rationales

    # Zero out direction/importance for irrelevant articles
    results.loc[results["relevance"] == 0, "direction"] = 0
    results.loc[results["relevance"] == 0, "importance"] = 0

    n_rel = (results["relevance"] == 1).sum()
    log.info(f"TF-IDF scoring complete: {n_rel}/{len(results)} relevant")

    # Drop intermediate columns
    results = results.drop(columns=["rel_confidence", "dir_confidence"], errors="ignore")

    return results
