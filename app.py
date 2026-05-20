"""
Streamlit Dashboard — NYT Green Subsidy Index
Acharya et al. (2025): Climate Transition Risks and the Energy Sector

Reproduces paper-style visualisations (Table 1, time-series, event studies)
and adds interactive controls for live exploration.

Run:  streamlit run app.py
"""

import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from plotly.subplots import make_subplots

from src.config import PipelineConfig, log
from src.fred_data import fetch_energy_prices
from src.index_builder import build_weekly_index
from src.regression import run_regression
from src.scorer_tfidf import score_articles_tfidf
from src.scraper import scrape_nyt

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Green Subsidy Index — Acharya et al. (2025)",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px; border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers: run async from Streamlit (sync context)
# ---------------------------------------------------------------------------
def _run_async(coro):
    """Run an async coroutine from Streamlit's sync context."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ===================================================================
# DATA LOADING
# ===================================================================
@st.cache_data
def load_csv_articles(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["date"])


@st.cache_data
def load_csv_index(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["week_start"])


def generate_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate realistic demo data when no pipeline output exists yet."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-03", "2024-12-30", freq="B")
    mask = rng.random(len(dates)) < 0.05
    article_dates = dates[mask]
    n = len(article_dates)
    directions = rng.choice([-1, 0, 1], size=n, p=[0.15, 0.25, 0.60])
    importances = rng.choice([1, 2, 3], size=n, p=[0.50, 0.35, 0.15])

    events = {
        "2022-08-16": ("Inflation Reduction Act signed", 1, 3),
        "2023-03-28": ("EU Green Deal Industrial Plan", 1, 3),
        "2023-06-07": ("Solar subsidy extension approved", 1, 2),
        "2024-01-15": ("IRA tax credits phase-out rumor", -1, 2),
        "2024-06-05": ("EU CBAM enforcement begins", 1, 2),
        "2024-11-06": ("Post-election subsidy uncertainty", -1, 3),
    }

    scored = pd.DataFrame({
        "date": article_dates,
        "headline": [f"Green subsidy news #{i}" for i in range(n)],
        "snippet": ["Sample snippet"] * n,
        "lead_paragraph": [""] * n,
        "section": rng.choice(["Business", "Climate", "Energy"], n),
        "relevance": np.ones(n, dtype=int),
        "direction": directions,
        "importance": importances,
        "rationale": ["demo"] * n,
    })
    event_rows = []
    for date_str, (headline, direction, importance) in events.items():
        event_rows.append({
            "date": pd.Timestamp(date_str), "headline": headline,
            "snippet": headline, "lead_paragraph": "",
            "section": "Climate", "relevance": 1,
            "direction": direction, "importance": importance,
            "rationale": "Key policy event",
        })
    scored = pd.concat([scored, pd.DataFrame(event_rows)], ignore_index=True)
    scored = scored.sort_values("date").reset_index(drop=True)
    weekly = build_weekly_index(scored)
    return scored, weekly


# ===================================================================
# SIDEBAR
# ===================================================================
st.sidebar.title("🌱 Configuration")

cfg = PipelineConfig()

# --- Data source selection ---
source_options = []
if cfg.has_nyt_key:
    source_options.append("🔴 NYT Live (API)")
source_options += ["📂 CSV existant", "🧪 Demo (synthetique)"]

data_source = st.sidebar.radio("Source de donnees", source_options, index=0)

scored_df = pd.DataFrame()
weekly_df = pd.DataFrame()

# ------------------------------------------------------------------
# NYT LIVE SCRAPING
# ------------------------------------------------------------------
if data_source == "🔴 NYT Live (API)":
    st.sidebar.success(f"Cle API NYT detectee")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Parametres de scraping")

    year_options = list(range(2013, 2026))
    selected_years = st.sidebar.multiselect(
        "Annees a scraper",
        year_options,
        default=[2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    )
    selected_months = st.sidebar.multiselect(
        "Mois (vide = tous)",
        list(range(1, 13)),
        default=[],
        format_func=lambda m: [
            "Jan", "Fev", "Mar", "Avr", "Mai", "Juin",
            "Juil", "Aout", "Sep", "Oct", "Nov", "Dec"
        ][m - 1],
    )

    scorer_choice = st.sidebar.radio(
        "Methode de scoring",
        ["TF-IDF + LR (CPU, recommande)", "Heuristique (mots-cles)"],
        index=0,
        help="TF-IDF+LR : classifieur entraine, CPU only.",
    )

    # --- Launch button ---
    if st.sidebar.button("🚀 Lancer le scraping NYT", type="primary", use_container_width=True):
        scrape_cfg = PipelineConfig(
            nyt_api_key=cfg.nyt_api_key,
            scrape_years=selected_years,
        )
        months = selected_months if selected_months else list(range(1, 13))

        # -- Stage 1: Scrape --
        with st.spinner("📥 Scraping NYT Archive API..."):
            progress = st.progress(0, text="Demarrage...")
            import aiohttp

            from src.filters import (passes_keyword_filter,
                                     passes_section_filter)

            async def scrape_with_progress():
                all_articles = []
                combos = [(y, m) for y in selected_years for m in months]
                total = len(combos)
                rate_limiter = asyncio.Semaphore(1)

                async with aiohttp.ClientSession() as session:
                    for idx, (year, month) in enumerate(combos):
                        progress.progress(
                            (idx) / total,
                            text=f"Telechargement {year}-{month:02d}  ({idx + 1}/{total})",
                        )

                        async with rate_limiter:
                            url = (
                                f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"
                                f"?api-key={scrape_cfg.nyt_api_key}"
                            )
                            try:
                                async with session.get(url, timeout=aiohttp.ClientTimeout(total=45)) as resp:
                                    if resp.status == 429:
                                        log.warning(f"Rate-limited {year}-{month:02d}, waiting 60s...")
                                        await asyncio.sleep(60)
                                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=45)) as retry:
                                            if retry.status != 200:
                                                continue
                                            data = await retry.json()
                                    elif resp.status != 200:
                                        log.error(f"HTTP {resp.status} for {year}-{month:02d}")
                                        continue
                                    else:
                                        data = await resp.json()
                            except Exception as e:
                                log.error(f"Error {year}-{month:02d}: {e}")
                                continue

                            docs = data.get("response", {}).get("docs", [])
                            for art in docs:
                                if not passes_section_filter(art):
                                    continue
                                headline = art.get("headline", {}).get("main", "")
                                snippet = art.get("snippet", "") or ""
                                lead = art.get("lead_paragraph", "") or ""
                                full_text = f"{headline} {snippet} {lead}"
                                if not passes_keyword_filter(full_text):
                                    continue
                                all_articles.append({
                                    "date": art.get("pub_date", ""),
                                    "headline": headline,
                                    "snippet": snippet,
                                    "lead_paragraph": lead,
                                    "section": art.get("section_name", ""),
                                    "news_desk": art.get("news_desk", ""),
                                    "web_url": art.get("web_url", ""),
                                    "text_to_analyze": full_text,
                                })

                            log.info(f"  {year}-{month:02d}: done ({len(all_articles)} cumul)")
                            await asyncio.sleep(scrape_cfg.nyt_rate_limit_delay)

                progress.progress(1.0, text="Scraping termine !")
                return all_articles

            raw_articles = _run_async(scrape_with_progress())

        if not raw_articles:
            st.error("Aucun article trouve. Verifiez votre cle API ou les filtres.")
        else:
            df_raw = pd.DataFrame(raw_articles)
            df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True, errors="coerce")
            df_raw = df_raw.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            st.success(f"✅ {len(df_raw)} articles pertinents recuperes !")

            # -- Stage 2: Scoring ---
            if scorer_choice.startswith("TF-IDF"):
                with st.spinner("🧠 Scoring TF-IDF + Logistic Regression (CPU)..."):
                    df_raw = score_articles_tfidf(df_raw)
                st.success("Scoring TF-IDF+LR termine.")
            else:
                # Heuristic fallback
                st.info("Scoring heuristique (mots-cles uniquement).")
                df_raw["relevance"] = 1
                directions = []
                importances = []
                for _, row in df_raw.iterrows():
                    txt = row["text_to_analyze"].lower()
                    pos_words = ["increase", "boost", "expand", "new", "sign", "approve",
                                 "extend", "launch", "invest", "billion", "fund"]
                    neg_words = ["cut", "repeal", "block", "reduce", "end", "cancel",
                                 "phase out", "eliminate", "halt", "oppose"]
                    pos = sum(1 for w in pos_words if w in txt)
                    neg = sum(1 for w in neg_words if w in txt)
                    directions.append(1 if pos > neg else (-1 if neg > pos else 0))
                    if any(w in txt for w in ["inflation reduction act", "ira ", "green deal",
                                               "trillion", "historic", "landmark"]):
                        importances.append(3)
                    elif any(w in txt for w in ["billion", "federal", "national", "major"]):
                        importances.append(2)
                    else:
                        importances.append(1)
                df_raw["direction"] = directions
                df_raw["importance"] = importances
                df_raw["rationale"] = "heuristic"

            # -- Stage 3: Build index & save --
            weekly = build_weekly_index(df_raw)

            # Save to CSV
            df_raw.to_csv(cfg.raw_csv, index=False)
            weekly.to_csv(cfg.index_csv, index=False)

            # Store in session state
            st.session_state["scored_df"] = df_raw
            st.session_state["weekly_df"] = weekly
            st.success(f"📊 Indice construit : {len(weekly)} semaines. CSV sauvegardes.")

    # Restore from session state
    if "scored_df" in st.session_state:
        scored_df = st.session_state["scored_df"]
        weekly_df = st.session_state["weekly_df"]
    else:
        # Try loading existing CSV
        if cfg.raw_csv.exists() and cfg.index_csv.exists():
            scored_df = load_csv_articles(str(cfg.raw_csv))
            weekly_df = load_csv_index(str(cfg.index_csv))
            st.sidebar.info("📂 Donnees precedentes chargees depuis CSV.")
        else:
            st.sidebar.warning("Cliquez sur 'Lancer le scraping' pour demarrer.")

# ------------------------------------------------------------------
# CSV MODE
# ------------------------------------------------------------------
elif data_source == "📂 CSV existant":
    raw_path = st.sidebar.text_input("Articles CSV", value="nyt_articles_raw.csv")
    idx_path = st.sidebar.text_input("Index CSV", value="weekly_green_subsidy_index.csv")
    scored_df = load_csv_articles(raw_path)
    weekly_df = load_csv_index(idx_path)
    if scored_df is None or weekly_df is None:
        st.sidebar.error("CSV introuvables.")
        scored_df = pd.DataFrame()
        weekly_df = pd.DataFrame()

# ------------------------------------------------------------------
# DEMO MODE
# ------------------------------------------------------------------
else:
    scored_df, weekly_df = generate_demo_data()

# --- Date filter ---
st.sidebar.markdown("---")
st.sidebar.subheader("Filtres temporels")
if weekly_df is not None and not weekly_df.empty:
    # Ensure timezone-naive AND time-stripped for date filter & merge with FRED
    if weekly_df["week_start"].dt.tz is not None:
        weekly_df["week_start"] = weekly_df["week_start"].dt.tz_localize(None)
    weekly_df["week_start"] = weekly_df["week_start"].dt.normalize()  # 12:34:30 → 00:00:00
    min_date = weekly_df["week_start"].min().date()
    max_date = weekly_df["week_start"].max().date()
    st.sidebar.caption(f"Donnees disponibles : {min_date} a {max_date}")
    date_range = st.sidebar.date_input(
        "Periode", value=(min_date, max_date),
        min_value=min_date, max_value=max_date,
    )
    if len(date_range) == 2:
        weekly_df = weekly_df[
            (weekly_df["week_start"].dt.date >= date_range[0])
            & (weekly_df["week_start"].dt.date <= date_range[1])
        ]

# ===================================================================
# HEADER
# ===================================================================
st.title("🌱 NYT Green Subsidy News Index")
st.markdown(
    "*Reproduction et extension empirique — Acharya et al. (2025): "
    "Climate Transition Risks and the Energy Sector*"
)

# Guard: no data yet
if weekly_df is None or weekly_df.empty or scored_df is None or scored_df.empty:
    st.warning(
        "Aucune donnee disponible. Utilisez la sidebar pour lancer le scraping NYT, "
        "charger un CSV, ou basculer en mode Demo."
    )
    st.stop()

st.markdown("---")

# ===================================================================
# KPI CARDS
# ===================================================================
n_articles = len(scored_df)
n_relevant = int((scored_df["relevance"] == 1).sum()) if "relevance" in scored_df.columns else n_articles
n_weeks = len(weekly_df)
avg_score = weekly_df["index_score"].mean()

cols = st.columns(4)
with cols[0]:
    st.metric("Articles analyses", f"{n_articles:,}")
with cols[1]:
    st.metric("Articles pertinents", f"{n_relevant:,}")
with cols[2]:
    st.metric("Semaines couvertes", n_weeks)
with cols[3]:
    st.metric("Score moyen hebdo", f"{avg_score:.2f}")

st.markdown("---")

# ===================================================================
# TABS
# ===================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Indice Hebdomadaire",
    "📊 Regression & Event Study",
    "🔬 Analyse des Articles",
    "📋 Table 1 (Reproduction)",
    "🧪 Sensibilite & Scenarios",
])

# -------------------------------------------------------------------
# TAB 1 — Weekly Index Time Series
# -------------------------------------------------------------------
with tab1:
    st.subheader("Indice Hebdomadaire des Subventions Vertes (NYT)")
    st.caption(
        "Index_w = Sigma(direction_i x importance_i) pour tous les articles pertinents de la semaine w"
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in weekly_df["index_score"]]
    fig.add_trace(
        go.Bar(
            x=weekly_df["week_start"], y=weekly_df["index_score"],
            marker_color=colors, name="Index Score", opacity=0.85,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=weekly_df["week_start"], y=weekly_df["n_articles"],
            mode="lines", name="Nb. articles",
            line=dict(color="#3498db", width=1.5), opacity=0.6,
            fill="tozeroy", fillcolor="rgba(52,152,219,0.08)",
        ),
        secondary_y=True,
    )

    if len(weekly_df) >= 8:
        rolling_4w = weekly_df["index_score"].rolling(4, center=True).mean()
        fig.add_trace(
            go.Scatter(
                x=weekly_df["week_start"], y=rolling_4w,
                mode="lines", name="Moyenne mobile 4 sem.",
                line=dict(color="#f39c12", width=2.5, dash="dash"),
            ),
            secondary_y=False,
        )

    # Align zeros: compute axis ranges so 0 is at the same vertical position
    idx_min = weekly_df["index_score"].min()
    idx_max = weekly_df["index_score"].max()
    y1_lo = idx_min * 1.15
    y1_hi = idx_max * 1.15
    zero_frac = (0 - y1_lo) / (y1_hi - y1_lo) if y1_hi != y1_lo else 0.5
    art_max = max(weekly_df["n_articles"].max(), 1) * 1.2
    y2_lo = -zero_frac / (1 - zero_frac) * art_max if zero_frac < 1 else 0

    fig.update_layout(
        height=500, template="plotly_white",
        legend=dict(orientation="h", y=-0.15),
        yaxis_title="Index Score (Sigma dir x imp)",
        yaxis2_title="Nombre d'articles",
        hovermode="x unified",
    )
    fig.update_yaxes(range=[y1_lo, y1_hi], secondary_y=False)
    fig.update_yaxes(
        range=[y2_lo, art_max], secondary_y=True,
        tickvals=list(range(0, int(art_max) + 1)),
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", secondary_y=False)
    st.plotly_chart(fig, width="stretch")

    # Cumulative index
    st.subheader("Indice Cumule (Signal de long terme)")
    cumulative = weekly_df["index_score"].cumsum()
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=weekly_df["week_start"], y=cumulative,
        fill="tozeroy", fillcolor="rgba(46,204,113,0.2)",
        line=dict(color="#27ae60", width=2),
        name="Indice cumule",
    ))
    fig_cum.update_layout(
        height=350, template="plotly_white",
        yaxis_title="Score cumule", hovermode="x unified",
    )
    st.plotly_chart(fig_cum, width="stretch")
    st.caption(
        "**Interpretation :** Une pente ascendante = flux net de news pro-subventions. "
        "Un aplatissement = retournement du sentiment politique."
    )

# -------------------------------------------------------------------
# TAB 2 — Regression with real FRED data
# -------------------------------------------------------------------
with tab2:
    st.subheader("Regression a haute frequence avec donnees FRED reelles")

    # --- Fetch FRED data (cached) ---
    @st.cache_data(ttl=3600, show_spinner="Chargement des prix FRED...")
    def _load_fred(api_key: str, start: str):
        from src.fred_data import fetch_energy_prices
        temp_cfg = PipelineConfig(fred_api_key=api_key)
        return fetch_energy_prices(temp_cfg, start=start)

    energy_df = pd.DataFrame()
    has_fred = cfg.has_fred_key

    if has_fred:
        energy_df = _load_fred(cfg.fred_api_key, "2012-01-01")

    if not energy_df.empty:
        st.success(f"Donnees FRED chargees : {len(energy_df)} semaines de prix reels (electricite + WTI)")

        # --- Price evolution chart ---
        st.markdown("#### Evolution des prix de l'energie (FRED)")
        fig_prices = make_subplots(specs=[[{"secondary_y": True}]])
        fig_prices.add_trace(go.Scatter(
            x=energy_df["week_start"], y=energy_df["elec_price"],
            mode="lines", name="Electricite ($/kWh)",
            line=dict(color="#e74c3c", width=1.5),
        ), secondary_y=False)
        fig_prices.add_trace(go.Scatter(
            x=energy_df["week_start"], y=energy_df["oil_price"],
            mode="lines", name="Petrole WTI ($/bbl)",
            line=dict(color="#2c3e50", width=1.5),
        ), secondary_y=True)
        # Add natural gas if available — scaled by 10 for visual comparability with oil
        if "gas_price" in energy_df.columns:
            fig_prices.add_trace(go.Scatter(
                x=energy_df["week_start"], y=energy_df["gas_price"] * 10,
                mode="lines", name="Gaz naturel ($/MMBtu x10)",
                line=dict(color="#16a085", width=1.5, dash="dot"),
            ), secondary_y=True)
        fig_prices.update_layout(
            height=350, template="plotly_white",
            legend=dict(orientation="h", y=-0.15),
            hovermode="x unified",
        )
        fig_prices.update_yaxes(title_text="Electricite ($/kWh)", secondary_y=False)
        fig_prices.update_yaxes(title_text="WTI ($/bbl) / Gaz ($/MMBtu x10)", secondary_y=True)
        st.plotly_chart(fig_prices, width="stretch")

        # --- Ratio chart ---
        st.markdown("#### Ratio Electricite / Petrole (indice de substitution)")
        fig_ratio = go.Figure()
        fig_ratio.add_trace(go.Scatter(
            x=energy_df["week_start"], y=energy_df["ratio_elec_oil"],
            fill="tozeroy", fillcolor="rgba(52,152,219,0.15)",
            line=dict(color="#2980b9", width=2),
            name="Ratio Elec/Oil",
        ))
        fig_ratio.update_layout(
            height=300, template="plotly_white",
            yaxis_title="Ratio (elec_price x 1000 / oil_price)",
            hovermode="x unified",
        )
        st.plotly_chart(fig_ratio, width="stretch")
        st.caption(
            "**Lecture :** Quand le ratio monte, l'electricite devient relativement plus chere "
            "que le petrole. Les subventions vertes devraient faire **baisser** ce ratio "
            "(electricite moins chere grace aux renouvelables)."
        )

        st.markdown("---")

    # --- Regression ---
    # Build target options dynamically (only show gas-based options if gas data is present)
    target_options = {}
    if not energy_df.empty and "ratio_gas_return" in energy_df.columns:
        target_options["Ratio Elec/Gaz naturel (recommande - canal direct)"] = "ratio_gas_return"
    target_options["Ratio Elec/Petrole WTI (legacy)"] = "ratio_return"
    target_options["Rendement Electricite seul"] = "elec_return"
    if not energy_df.empty and "gas_return" in energy_df.columns:
        target_options["Rendement Gaz naturel seul"] = "gas_return"
    target_options["Rendement Petrole WTI seul"] = "oil_return"

    target_label = st.selectbox(
        "Variable dependante de la regression",
        list(target_options.keys()),
        index=0,
        help=(
            "Le ratio Elec/Gaz est theoriquement superieur : le gaz naturel "
            "fournit ~40% de l'electricite US, contre ~0.5% pour le petrole. "
            "C'est donc le vrai canal de substitution teste par les subventions vertes."
        ),
    )
    target_col = target_options[target_label]

    # Run regression
    if not energy_df.empty:
        reg_results, weekly_reg = run_regression(weekly_df, energy_df, target_col=target_col)
        data_source_label = "Donnees reelles FRED"
    else:
        if cfg.has_fred_key:
            st.error(
                "⚠️ **FRED API injoignable** (timeout reseau ou panne). "
                "La regression ci-dessous utilise des **donnees synthetiques calibrees** "
                "(beta = -0.003 baked in). Les coefficients et p-values affichees "
                "ne sont **PAS** issus de donnees reelles. Recharge la page dans quelques minutes."
            )
        else:
            st.warning("Cle FRED_API_KEY absente du .env — regression sur donnees simulees.")
        reg_results, weekly_reg = run_regression(weekly_df, None, target_col=target_col)
        data_source_label = "⚠️ DONNEES SIMULEES (FRED indisponible)"

    if "error" not in reg_results and target_col in weekly_reg.columns:
        st.markdown(f"#### Regression : `{target_label}` = alpha + beta x Index_score")
        st.caption(f"Source : {data_source_label} | Ecarts-types robustes HC1")

        col1, col2 = st.columns([1, 1])

        with col1:
            fig_scat = go.Figure()
            fig_scat.add_trace(go.Scatter(
                x=weekly_reg["index_score"], y=weekly_reg[target_col],
                mode="markers",
                marker=dict(size=6, color="#2c3e50", opacity=0.5),
                name="Observations",
            ))
            x_range = np.linspace(
                weekly_reg["index_score"].min(),
                weekly_reg["index_score"].max(), 100,
            )
            y_fit = reg_results["alpha"] + reg_results["beta"] * x_range
            fig_scat.add_trace(go.Scatter(
                x=x_range, y=y_fit, mode="lines",
                line=dict(color="red", width=2.5, dash="dash"),
                name=f"OLS: beta = {reg_results['beta']:.6f}",
            ))
            fig_scat.update_layout(
                height=450, template="plotly_white",
                xaxis_title="Index Score hebdo (S_t)",
                yaxis_title=f"Rendement hebdo ({target_label})",
                title="News de subventions vs. prix de l'energie",
            )
            fig_scat.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_scat.add_vline(x=0, line_dash="dot", line_color="gray")
            st.plotly_chart(fig_scat, width="stretch")

        with col2:
            st.markdown("#### Resultats OLS")
            results_table = pd.DataFrame({
                "Parametre": ["alpha (constante)", "beta (index)", "R-carre",
                              "R-carre ajuste", "N obs.", "F-stat"],
                "Valeur": [
                    f"{reg_results['alpha']:.6f}",
                    f"{reg_results['beta']:.6f}",
                    f"{reg_results['r_squared']:.4f}",
                    f"{reg_results['adj_r_squared']:.4f}",
                    f"{reg_results['n_obs']}",
                    f"{reg_results['f_stat']:.2f}",
                ],
                "Ecart-type": [
                    "—",
                    f"{reg_results['beta_se']:.6f}",
                    "—", "—", "—", "—",
                ],
                "t-stat": [
                    "—",
                    f"{reg_results['beta_tstat']:.2f}",
                    "—", "—", "—", "—",
                ],
                "p-value": [
                    "—",
                    f"{reg_results['beta_pvalue']:.4f}",
                    "—", "—", "—", "—",
                ],
            })
            st.dataframe(results_table, width="stretch", hide_index=True)

            # Significance badge
            p = reg_results["beta_pvalue"]
            if p < 0.01:
                st.success(f"*** Significatif a 1% (p = {p:.4f})")
            elif p < 0.05:
                st.success(f"** Significatif a 5% (p = {p:.4f})")
            elif p < 0.10:
                st.warning(f"* Significatif a 10% (p = {p:.4f})")
            else:
                st.error(f"Non significatif (p = {p:.4f})")

            st.info(f"**Interpretation :** {reg_results['interpretation']}")

        # --- Economic magnitude panel (per reviewer feedback) ---
        st.markdown("---")
        st.subheader("💰 Magnitude economique du coefficient")
        st.caption(
            "Le coefficient beta seul n'est pas parlant. Cette section traduit "
            "beta en effets tangibles sur le marche."
        )

        mag_cols = st.columns(4)
        with mag_cols[0]:
            st.metric(
                "Effet d'un choc 1 ecart-type",
                f"{reg_results['effect_1sd_pct']:+.3f} %",
                help=(
                    f"Si l'indice de news augmente de 1 ecart-type "
                    f"({reg_results['sd_index']:.2f} points), la variable cible "
                    f"varie de {reg_results['effect_1sd_pct']:+.3f} % la meme semaine."
                ),
            )
        with mag_cols[1]:
            st.metric(
                "Effet d'un choc type IRA",
                f"{reg_results['effect_ira_pct']:+.3f} %",
                help=(
                    "Une annonce 'direction = +1, importance = 3' "
                    "(equivalent IRA, Green Deal, Clean Power Plan) "
                    "produit cette variation hebdomadaire."
                ),
            )
        with mag_cols[2]:
            st.metric(
                "Effet annualise (+1/sem.)",
                f"{reg_results['effect_annualized_pct']:+.2f} %",
                help=(
                    "Si l'indice etait soutenu a +1 chaque semaine pendant 1 an, "
                    "l'effet cumule (compose) sur 52 semaines serait celui-ci."
                ),
            )
        with mag_cols[3]:
            stn = reg_results["signal_to_noise"]
            st.metric(
                "Ratio signal/bruit",
                f"{stn:.3f}" if not np.isnan(stn) else "n/a",
                help=(
                    "Part de la volatilite hebdomadaire de la cible "
                    "expliquee par un choc d'1 ecart-type sur l'indice. "
                    "Une valeur > 0.10 = signal significatif vs bruit de marche."
                ),
            )

        # Concrete dollar translation if elec is in the target
        if target_col in ("ratio_return", "ratio_gas_return", "elec_return") and not energy_df.empty:
            avg_elec = energy_df["elec_price"].mean()
            # Typical US household: 877 kWh/month (EIA) → ~10,500 kWh/year
            annual_kwh = 10_500
            annual_bill = avg_elec * annual_kwh
            bill_change_ira = annual_bill * reg_results["effect_ira_pct"] / 100
            bill_change_annual = annual_bill * reg_results["effect_annualized_pct"] / 100

            st.markdown(
                f"**Traduction concrete (menage US moyen, {annual_kwh:,} kWh/an, "
                f"facture de base ≈ ${annual_bill:,.0f}/an) :**"
            )
            st.markdown(
                f"- Choc IRA isole → variation de facture annuelle : **${bill_change_ira:+,.2f}**\n"
                f"- Subventions soutenues toute l'annee → variation : **${bill_change_annual:+,.2f}**"
            )

        # Magnitude diagnostic: only meaningful if p < 0.10
        pval = reg_results["beta_pvalue"]
        if pval > 0.10:
            st.error(
                f"❌ **AUCUN SIGNAL DETECTABLE** (p = {pval:.3f} >> 0.10). "
                "Les 'magnitudes' affichees ci-dessus sont **SANS SIGNIFICATION STATISTIQUE**. "
                "Il n'y a rien à interpréter: les news de subventions du NYT n'impactent pas "
                "mesurablement les prix/ratios d'énergie à l'horizon hebdomadaire. "
                "\n\n**Pourquoi?** Les marchés énergétiques mondiaux sont dominés par: "
                "(1) décisions OPEP, (2) géopolitique/guerres, (3) demande macro globale. "
                "Un signal médiatique américain est un bruit blanc negligeable. "
                "C'est un **résultat négatif honnête**, pas un bug."
            )
        else:
            # Only diagnose magnitude if we actually have a detected signal
            abs_ira = abs(reg_results["effect_ira_pct"])
            if abs_ira < 0.1:
                st.info(
                    f"🔍 **Effet statistiquement significatif mais economiquement modeste** "
                    f"({abs_ira:.3f}% par choc IRA). Le signal existe mais la magnitude "
                    "est ecrasee par d'autres facteurs (OPEP, geopolitique, demande)."
                )
            else:
                st.success(
                    f"✅ **Effet economiquement substantiel** ({abs_ira:.2f}% par choc IRA). "
                    "La magnitude est compatible avec un impact reel des politiques climatiques "
                    "sur les prix de l'energie."
                )

        # --- Rolling beta ---
        st.markdown("---")
        st.subheader("Stabilite du coefficient beta (fenetre glissante)")
        window = st.slider("Taille de la fenetre (semaines)", 12, 52, 26)

        if len(weekly_reg) >= window:
            rolling_betas = []
            rolling_dates = []
            for i in range(window, len(weekly_reg)):
                chunk = weekly_reg.iloc[i - window:i]
                X_r = sm.add_constant(chunk["index_score"])
                y_r = chunk[target_col]
                try:
                    m = sm.OLS(y_r, X_r).fit()
                    rolling_betas.append(m.params.get("index_score", np.nan))
                    rolling_dates.append(chunk["week_start"].iloc[-1])
                except Exception:
                    rolling_betas.append(np.nan)
                    rolling_dates.append(chunk["week_start"].iloc[-1])

            fig_rb = go.Figure()
            fig_rb.add_trace(go.Scatter(
                x=rolling_dates, y=rolling_betas,
                mode="lines", line=dict(color="#8e44ad", width=2),
                name=f"Beta glissant ({window} sem.)",
            ))
            fig_rb.add_hline(y=0, line_dash="dash", line_color="gray",
                             annotation_text="beta = 0 (aucun effet)")
            fig_rb.update_layout(
                height=350, template="plotly_white",
                yaxis_title="beta estime",
                title=f"Evolution temporelle de beta (fenetre = {window} semaines)",
            )
            st.plotly_chart(fig_rb, width="stretch")
            st.caption(
                "**Lecture :** Un beta negatif persistant signifie que les subventions "
                "reduisent durablement le cout relatif de l'electricite. Un basculement "
                "vers un beta positif signale un changement structurel des anticipations."
            )
    else:
        st.error(f"Regression impossible : {reg_results.get('error', 'unknown')}")

# -------------------------------------------------------------------
# TAB 3 — Article-level analysis
# -------------------------------------------------------------------
with tab3:
    st.subheader("Exploration des articles scores")

    if "relevance" in scored_df.columns:
        rel = scored_df[scored_df["relevance"] == 1].copy()
    else:
        rel = scored_df.copy()

    col1, col2, col3 = st.columns(3)
    with col1:
        if "direction" in rel.columns:
            dir_counts = rel["direction"].value_counts().reindex([-1, 0, 1], fill_value=0)
            fig_dir = go.Figure(go.Bar(
                x=["Baisse (-1)", "Neutre (0)", "Hausse (+1)"],
                y=dir_counts.values,
                marker_color=["#e74c3c", "#95a5a6", "#2ecc71"],
            ))
            fig_dir.update_layout(
                title="Distribution de la direction", height=300,
                template="plotly_white", yaxis_title="Nb. articles",
            )
            st.plotly_chart(fig_dir, width="stretch")

    with col2:
        if "importance" in rel.columns:
            imp_counts = rel["importance"].value_counts().reindex([1, 2, 3], fill_value=0)
            fig_imp = go.Figure(go.Bar(
                x=["Mineur (1)", "Majeur (2)", "Massif (3)"],
                y=imp_counts.values,
                marker_color=["#f1c40f", "#e67e22", "#e74c3c"],
            ))
            fig_imp.update_layout(
                title="Distribution de l'importance", height=300,
                template="plotly_white", yaxis_title="Nb. articles",
            )
            st.plotly_chart(fig_imp, width="stretch")

    with col3:
        if "section" in rel.columns:
            sec_counts = rel["section"].value_counts().head(8)
            fig_sec = go.Figure(go.Pie(
                labels=sec_counts.index, values=sec_counts.values, hole=0.4,
            ))
            fig_sec.update_layout(title="Sections NYT", height=300)
            st.plotly_chart(fig_sec, width="stretch")

    # Monthly heatmap
    if "direction" in rel.columns and "importance" in rel.columns:
        st.markdown("---")
        st.subheader("Heatmap mensuelle : intensite du signal")
        rel_copy = rel.copy()
        rel_copy["score"] = rel_copy["direction"] * rel_copy["importance"]
        month_source = rel_copy["date"]
        if getattr(month_source.dt, "tz", None) is not None:
            month_source = month_source.dt.tz_convert(None)
        rel_copy["month"] = month_source.dt.to_period("M").astype(str)
        monthly = rel_copy.groupby("month").agg(
            total_score=("score", "sum"),
            count=("score", "count"),
        ).reset_index()
        monthly["month_dt"] = pd.to_datetime(monthly["month"])
        monthly["year"] = monthly["month_dt"].dt.year
        monthly["month_name"] = monthly["month_dt"].dt.strftime("%b")

        pivot = monthly.pivot_table(index="year", columns="month_name",
                                     values="total_score", fill_value=0)
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])

        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=[str(y) for y in pivot.index],
            colorscale="RdYlGn", zmid=0,
            colorbar_title="Score",
        ))
        fig_heat.update_layout(
            height=300, template="plotly_white",
            title="Score mensuel (vert = pro-subvention, rouge = anti-subvention)",
        )
        st.plotly_chart(fig_heat, width="stretch")

    # Article table
    st.markdown("---")
    st.subheader("Articles les plus impactants")
    display_cols = [c for c in ["date", "headline", "direction", "importance", "rationale", "web_url"]
                    if c in rel.columns]
    if "direction" in rel.columns and "importance" in rel.columns:
        rel_sorted = rel.copy()
        rel_sorted["abs_score"] = (rel_sorted["direction"] * rel_sorted["importance"]).abs()
        top = rel_sorted.nlargest(20, "abs_score")[display_cols]
    else:
        top = rel.head(20)[display_cols]
    st.dataframe(top, width="stretch", hide_index=True)

# -------------------------------------------------------------------
# TAB 4 — Table 1 Reproduction
# -------------------------------------------------------------------
with tab4:
    st.subheader("Reproduction du Tableau 1 : Impact des news sur les prix de l'energie")
    st.caption("Adapte d'Acharya et al. (2025), Table 1 — etendu avec la categorie Subventions Vertes")

    table1_data = {
        "Categorie de News": [
            "Climate policy (carbon tax)",
            "Technology breakthrough",
            "Fossil fuel demand shock",
            "Geopolitical supply shock",
            "🌱 Green Subsidies (extension)",
        ],
        "Proxy dans le modele": [
            "Hausse de tau (taxe carbone)",
            "Hausse de p_BT (proba. breakthrough)",
            "Choc sur D (demande d'energie)",
            "Choc sur offre S (OPEP, conflit)",
            "Hausse de sigma (taux de subvention)",
        ],
        "Effet sur P_0 (prix spot)": [
            "Negatif (Green Paradox)",
            "Negatif",
            "Positif",
            "Positif",
            "Ambigu (GP vs Fossilflation)",
        ],
        "Effet sur investissement fossile": [
            "Reduction", "Reduction", "Hausse",
            "Neutre / Hausse", "Reduction forte (Entrants)",
        ],
        "Canal theorique": [
            "Extraction acceleree par anticipation fiscale",
            "Substitution attendue -> actifs echoues",
            "Hausse de la demande courante",
            "Reduction exogene de l'offre",
            "Subvention -> competitivite verte -> gel invest. fossile",
        ],
    }
    st.dataframe(pd.DataFrame(table1_data), width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("""
    #### Interpretation
    - **Green Paradox (beta < 0)** : L'incumbent accelere l'extraction aujourd'hui,
      car la valeur future de ses reserves diminue. Cela **fait baisser** le prix spot.
    - **Fossilflation (beta > 0)** : Les subventions sont si massives qu'elles decouragent
      tout nouvel investissement fossile. L'incumbent anticipe une penurie et **retient**
      son offre, faisant **monter** le prix.

    La direction du coefficient beta depend du **ratio entre l'effet de substitution**
    (qui pousse a extraire vite) et **l'effet d'investissement** (qui reduit l'offre future).
    """)

# -------------------------------------------------------------------
# TAB 5 — Sensitivity: 3D Heatmap (beta x noise -> p-value)
# -------------------------------------------------------------------
with tab5:
    st.subheader("Analyse de sensibilite : Detectabilite du signal climatique")
    st.markdown(
        "Cette simulation Monte Carlo repond a la question fondamentale : "
        "**pour quelles combinaisons de beta (sensibilite au choc) et de bruit "
        "de marche (sigma_epsilon) le signal des subventions vertes est-il "
        "statistiquement detectable ?**"
    )

    rng = np.random.default_rng(42)
    n = len(weekly_df)
    idx_scores = weekly_df["index_score"].values

    # --- Grid parameters ---
    st.markdown("#### Parametres de la grille de simulation")
    col_g1, col_g2, col_g3 = st.columns(3)
    with col_g1:
        beta_min = st.number_input("beta min", value=-0.05, step=0.01, format="%.3f")
        beta_max = st.number_input("beta max", value=0.05, step=0.01, format="%.3f")
    with col_g2:
        noise_min = st.number_input("sigma min", value=0.005, step=0.005, format="%.3f")
        noise_max = st.number_input("sigma max", value=0.10, step=0.01, format="%.3f")
    with col_g3:
        grid_res = st.slider("Resolution de la grille", 15, 50, 30)
        n_simulations = st.slider("Simulations par cellule", 1, 50, 10)

    z_metric = st.radio(
        "Metrique de l'axe Z",
        ["p-value du coefficient beta", "R-carre"],
        index=0, horizontal=True,
    )

    betas_grid = np.linspace(beta_min, beta_max, grid_res)
    noises_grid = np.linspace(noise_min, noise_max, grid_res)

    # --- Monte Carlo simulation ---
    with st.spinner("Simulation Monte Carlo en cours..."):
        Z = np.zeros((len(noises_grid), len(betas_grid)))

        for i, sigma in enumerate(noises_grid):
            for j, beta in enumerate(betas_grid):
                metrics = []
                for sim in range(n_simulations):
                    y_sim = beta * idx_scores + rng.normal(0, sigma, n)
                    X_sim = sm.add_constant(idx_scores)
                    m = sm.OLS(y_sim, X_sim).fit()
                    if z_metric.startswith("p-value"):
                        val = float(m.pvalues[1]) if len(m.pvalues) > 1 else 1.0
                    else:
                        val = m.rsquared
                    metrics.append(val)
                Z[i, j] = np.mean(metrics)

    # ---- HEATMAP 2D ----
    st.markdown("---")
    st.markdown("#### Heatmap 2D : Zone de detection statistique")

    if z_metric.startswith("p-value"):
        # Clip for visual clarity
        Z_display = np.clip(Z, 0, 1)
        colorscale = [
            [0.0, "#1a9850"],    # p ~ 0 : highly significant (green)
            [0.05, "#91cf60"],   # p = 0.05 : threshold
            [0.10, "#fee08b"],   # p = 0.10 : marginal
            [0.30, "#fc8d59"],   # p = 0.30 : weak
            [1.0, "#d73027"],    # p = 1.0 : no signal (red)
        ]
        zmid = 0.05
        colorbar_title = "p-value"
    else:
        Z_display = Z
        colorscale = "Viridis"
        zmid = None
        colorbar_title = "R-carre"

    fig_heatmap = go.Figure(go.Heatmap(
        z=Z_display,
        x=np.round(betas_grid, 4),
        y=np.round(noises_grid, 4),
        colorscale=colorscale,
        zmid=zmid,
        colorbar_title=colorbar_title,
        hovertemplate=(
            "beta: %{x:.4f}<br>"
            "sigma: %{y:.4f}<br>"
            + colorbar_title + ": %{z:.4f}<extra></extra>"
        ),
    ))

    # Add significance contour line at p=0.05
    if z_metric.startswith("p-value"):
        fig_heatmap.add_contour(
            z=Z_display,
            x=np.round(betas_grid, 4),
            y=np.round(noises_grid, 4),
            contours=dict(
                start=0.05, end=0.05, size=0,
                coloring="none",
                showlabels=True,
                labelfont=dict(size=12, color="white"),
            ),
            line=dict(color="white", width=2, dash="dash"),
            showscale=False,
            name="p = 0.05",
        )

    fig_heatmap.update_layout(
        height=550, template="plotly_white",
        title=f"Detectabilite : {colorbar_title} en fonction de beta et du bruit (N={n} semaines)",
        xaxis_title="beta (sensibilite du petrole aux news de subventions)",
        yaxis_title="sigma_epsilon (volatilite exogene du marche)",
    )
    # Mark Green Paradox / Fossilflation zones
    fig_heatmap.add_vline(x=0, line_dash="dot", line_color="white", opacity=0.5)
    fig_heatmap.add_annotation(
        x=beta_min * 0.6, y=noise_max * 0.9,
        text="GREEN PARADOX<br>(beta < 0)",
        showarrow=False, font=dict(color="white", size=11),
    )
    fig_heatmap.add_annotation(
        x=beta_max * 0.6, y=noise_max * 0.9,
        text="FOSSILFLATION<br>(beta > 0)",
        showarrow=False, font=dict(color="white", size=11),
    )
    st.plotly_chart(fig_heatmap, width="stretch")

    st.caption(
        "**Lecture :** La zone **verte** (p < 0.05) represente les combinaisons ou le signal "
        "climatique est statistiquement detectable. La zone **rouge** (p >> 0.05) est la "
        "zone d'aveuglement ou le bruit de marche noie le signal. "
        "**C'est pourquoi Acharya et al. utilisent la haute frequence** : en reduisant sigma, "
        "on elargit la zone verte de detection."
    )

    # ---- SURFACE 3D ----
    st.markdown("---")
    st.markdown("#### Surface 3D : Paysage de detectabilite")

    fig_3d = go.Figure(go.Surface(
        z=Z_display,
        x=np.round(betas_grid, 4),
        y=np.round(noises_grid, 4),
        colorscale="RdYlGn_r" if z_metric.startswith("p-value") else "Viridis",
        colorbar_title=colorbar_title,
        opacity=0.9,
    ))

    # Add p=0.05 plane if p-value mode
    if z_metric.startswith("p-value"):
        plane_z = np.full_like(Z_display, 0.05)
        fig_3d.add_trace(go.Surface(
            z=plane_z,
            x=np.round(betas_grid, 4),
            y=np.round(noises_grid, 4),
            colorscale=[[0, "rgba(255,165,0,0.3)"], [1, "rgba(255,165,0,0.3)"]],
            showscale=False, name="Seuil p=0.05",
            opacity=0.4,
        ))

    fig_3d.update_layout(
        height=600,
        title=f"Surface de {colorbar_title} : beta x sigma → detectabilite",
        scene=dict(
            xaxis_title="beta",
            yaxis_title="sigma_epsilon",
            zaxis_title=colorbar_title,
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
        ),
    )
    st.plotly_chart(fig_3d, width="stretch")

    st.markdown("""
    #### Interpretation economique pour le jury

    Ce graphique 3D materialise le concept central d'Acharya et al. (2025) :

    - **Crete gauche (beta < 0)** : regime de *Green Paradox* — les subventions poussent
      les firmes fossiles a extraire plus vite, faisant baisser le prix spot.
    - **Crete droite (beta > 0)** : regime de *Fossilflation* — les subventions gelent
      l'investissement fossile, provoquant une retention d'offre et une hausse des prix.
    - **Vallee centrale (beta ≈ 0)** : aucun signal detectable, quelle que soit la volatilite.
    - **Quand sigma augmente** (bruit de marche) : les cretes s'aplatissent, le signal
      climatique est noye. C'est precisement ce qui se produit a basse frequence (mensuelle).

    **Conclusion :** la haute frequence (hebdomadaire) est indispensable pour detecter
    empiriquement l'effet des subventions vertes sur les prix de l'energie.
    """)

# ===================================================================
# FOOTER
# ===================================================================
st.markdown("---")
st.markdown(
    "<center><small>NYT Green Subsidy Index Pipeline — "
    "Master 2 ETEEN 2025-2026 — Acharya et al. (2025) Extension<br>"
    "Auteur : Loic NEMBOT | Local AI Stack (Docker/n8n)</small></center>",
    unsafe_allow_html=True,
)
