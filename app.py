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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import statsmodels.api as sm
import streamlit as st

from src.config import PipelineConfig, log
from src.scraper import scrape_nyt
from src.scorer import score_articles
from src.index_builder import build_weekly_index
from src.regression import run_regression

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
    st.sidebar.success(f"Cle API NYT detectee ({cfg.nyt_api_key[:8]}...)")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Parametres de scraping")

    year_options = list(range(2018, 2027))
    selected_years = st.sidebar.multiselect(
        "Annees a scraper",
        year_options,
        default=[2022, 2023, 2024],
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

    use_llm = st.sidebar.checkbox(
        "Scorer via Local AI Stack (port 9485)",
        value=False,
        help="Necessite que votre Docker Local AI Stack soit actif.",
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
            from src.filters import passes_section_filter, passes_keyword_filter

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

            # -- Stage 2: LLM scoring (optional) --
            if use_llm:
                with st.spinner("🧠 Scoring via Local AI Stack..."):
                    df_raw = _run_async(score_articles(df_raw, scrape_cfg))
            else:
                # Simple heuristic scoring without LLM
                st.info("💡 Scoring heuristique (sans LLM). Activez le scoring LLM pour plus de precision.")
                from src.filters import passes_keyword_filter
                from src.config import SUBSIDY_PATTERNS, ENERGY_PATTERNS

                df_raw["relevance"] = 1  # already keyword-filtered
                directions = []
                importances = []
                for _, row in df_raw.iterrows():
                    txt = row["text_to_analyze"].lower()
                    # Direction heuristic
                    pos_words = ["increase", "boost", "expand", "new", "sign", "approve",
                                 "extend", "launch", "invest", "billion", "fund"]
                    neg_words = ["cut", "repeal", "block", "reduce", "end", "cancel",
                                 "phase out", "eliminate", "halt", "oppose"]
                    pos = sum(1 for w in pos_words if w in txt)
                    neg = sum(1 for w in neg_words if w in txt)
                    if pos > neg:
                        directions.append(1)
                    elif neg > pos:
                        directions.append(-1)
                    else:
                        directions.append(0)

                    # Importance heuristic
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

# --- Regression params ---
st.sidebar.markdown("---")
st.sidebar.subheader("Parametres de regression")
synthetic_beta = st.sidebar.slider(
    "Beta synthetique (Green Paradox < 0 / Fossilflation > 0)",
    min_value=-0.05, max_value=0.05, value=-0.005, step=0.001, format="%.3f",
)
noise_std = st.sidebar.slider("Ecart-type du bruit", 0.005, 0.10, 0.03, 0.005)

# --- Date filter ---
st.sidebar.markdown("---")
st.sidebar.subheader("Filtres temporels")
if weekly_df is not None and not weekly_df.empty:
    min_date = weekly_df["week_start"].min().date()
    max_date = weekly_df["week_start"].max().date()
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

    fig.update_layout(
        height=500, template="plotly_white",
        legend=dict(orientation="h", y=-0.15),
        yaxis_title="Index Score (Sigma dir x imp)",
        yaxis2_title="Nombre d'articles",
        hovermode="x unified",
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
# TAB 2 — Regression & Event Study
# -------------------------------------------------------------------
with tab2:
    st.subheader("Regression a haute frequence : Delta P_oil = alpha + beta x Index + epsilon")

    rng = np.random.default_rng(42)
    n = len(weekly_df)
    weekly_reg = weekly_df.copy()
    weekly_reg["oil_return"] = (
        synthetic_beta * weekly_reg["index_score"]
        + rng.normal(0, noise_std, n)
    )

    X = sm.add_constant(weekly_reg["index_score"])
    y = weekly_reg["oil_return"]
    model = sm.OLS(y, X).fit(cov_type="HC1")

    col1, col2 = st.columns([1, 1])

    with col1:
        fig_scat = go.Figure()
        fig_scat.add_trace(go.Scatter(
            x=weekly_reg["index_score"], y=weekly_reg["oil_return"],
            mode="markers",
            marker=dict(size=6, color="#2c3e50", opacity=0.5),
            name="Observations",
        ))
        x_range = np.linspace(
            weekly_reg["index_score"].min(),
            weekly_reg["index_score"].max(), 100,
        )
        y_fit = model.params["const"] + model.params["index_score"] * x_range
        fig_scat.add_trace(go.Scatter(
            x=x_range, y=y_fit, mode="lines",
            line=dict(color="red", width=2.5, dash="dash"),
            name=f"OLS: beta = {model.params['index_score']:.4f}",
        ))
        fig_scat.update_layout(
            height=450, template="plotly_white",
            xaxis_title="Index Score hebdo",
            yaxis_title="Rendement petrole (Delta P/P)",
            title="Green Subsidy News vs. Oil Returns",
        )
        fig_scat.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_scat.add_vline(x=0, line_dash="dot", line_color="gray")
        st.plotly_chart(fig_scat, width="stretch")

    with col2:
        st.markdown("#### Resultats OLS (Ecarts-types robustes HC1)")
        interpretation = (
            "**Green Paradox** : les firmes fossiles accelerent l'extraction"
            if model.params["index_score"] < 0
            else "**Fossilflation** : gel de l'investissement -> retention d'offre"
        )
        results_table = pd.DataFrame({
            "Parametre": ["alpha (constante)", "beta (index)", "R-carre",
                          "R-carre ajuste", "N obs.", "F-stat"],
            "Valeur": [
                f"{model.params['const']:.6f}",
                f"{model.params['index_score']:.6f}",
                f"{model.rsquared:.4f}",
                f"{model.rsquared_adj:.4f}",
                f"{int(model.nobs)}",
                f"{model.fvalue:.2f}",
            ],
            "Ecart-type": [
                f"{model.bse['const']:.6f}",
                f"{model.bse['index_score']:.6f}",
                "—", "—", "—", "—",
            ],
            "t-stat": [
                f"{model.tvalues['const']:.2f}",
                f"{model.tvalues['index_score']:.2f}",
                "—", "—", "—", "—",
            ],
            "p-value": [
                f"{model.pvalues['const']:.4f}",
                f"{model.pvalues['index_score']:.4f}",
                "—", "—", "—", "—",
            ],
        })
        st.dataframe(results_table, width="stretch", hide_index=True)
        st.info(f"Interpretation : {interpretation}")

    # Rolling beta
    st.markdown("---")
    st.subheader("Stabilite du coefficient beta (fenetre glissante)")
    window = st.slider("Taille de la fenetre (semaines)", 12, 52, 26)

    if len(weekly_reg) >= window:
        rolling_betas = []
        rolling_dates = []
        for i in range(window, len(weekly_reg)):
            chunk = weekly_reg.iloc[i - window:i]
            X_r = sm.add_constant(chunk["index_score"])
            y_r = chunk["oil_return"]
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
                         annotation_text="Seuil Green Paradox / Fossilflation")
        fig_rb.update_layout(
            height=350, template="plotly_white",
            yaxis_title="beta estime",
            title=f"Evolution temporelle de beta (fenetre = {window} semaines)",
        )
        st.plotly_chart(fig_rb, width="stretch")
        st.caption(
            "**Lecture :** Quand beta passe de negatif a positif, le marche "
            "bascule du regime Green Paradox au regime Fossilflation."
        )

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
        rel_copy["month"] = rel_copy["date"].dt.to_period("M").astype(str)
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
# TAB 5 — Sensitivity & Scenarios
# -------------------------------------------------------------------
with tab5:
    st.subheader("Analyse de sensibilite : beta en fonction des parametres")

    st.markdown("#### Scenario 1 : Variation du beta theorique")
    betas_range = np.linspace(-0.05, 0.05, 50)
    rng = np.random.default_rng(42)
    n = len(weekly_df)
    idx_scores = weekly_df["index_score"].values
    r_squareds = []
    p_values_list = []

    for b in betas_range:
        y_sim = b * idx_scores + rng.normal(0, noise_std, n)
        X_sim = sm.add_constant(idx_scores)
        m = sm.OLS(y_sim, X_sim).fit()
        r_squareds.append(m.rsquared)
        p_values_list.append(float(m.pvalues[1]) if len(m.pvalues) > 1 else 1.0)

    fig_sens = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sens.add_trace(go.Scatter(
        x=betas_range, y=r_squareds,
        mode="lines", name="R-carre",
        line=dict(color="#2ecc71", width=2),
    ), secondary_y=False)
    fig_sens.add_trace(go.Scatter(
        x=betas_range, y=p_values_list,
        mode="lines", name="p-value (beta)",
        line=dict(color="#e74c3c", width=2, dash="dash"),
    ), secondary_y=True)
    fig_sens.add_hline(y=0.05, line_dash="dot", line_color="orange",
                       annotation_text="Seuil 5%", secondary_y=True)
    fig_sens.add_vline(x=0, line_dash="dot", line_color="gray")
    fig_sens.update_layout(
        height=400, template="plotly_white",
        title="Pouvoir statistique en fonction de beta",
        xaxis_title="beta theorique",
    )
    fig_sens.update_yaxes(title_text="R-carre", secondary_y=False)
    fig_sens.update_yaxes(title_text="p-value", secondary_y=True)
    st.plotly_chart(fig_sens, width="stretch")

    st.caption(
        "**Lecture :** Plus |beta| est grand, plus le R-carre augmente et la p-value "
        "diminue. La zone ou p < 0.05 definit le seuil de detectabilite."
    )

    # Scenario 2: Noise sensitivity
    st.markdown("---")
    st.markdown("#### Scenario 2 : Impact du bruit de marche")
    noise_range = np.arange(0.005, 0.10, 0.005)
    detect = []
    for ns in noise_range:
        y_sim = synthetic_beta * idx_scores + rng.normal(0, ns, n)
        X_sim = sm.add_constant(idx_scores)
        m = sm.OLS(y_sim, X_sim).fit()
        detect.append(float(m.pvalues[1]) if len(m.pvalues) > 1 else 1.0)

    fig_noise = go.Figure()
    fig_noise.add_trace(go.Scatter(
        x=noise_range, y=detect,
        mode="lines+markers",
        line=dict(color="#8e44ad", width=2),
        name="p-value de beta",
    ))
    fig_noise.add_hline(y=0.05, line_dash="dot", line_color="orange",
                        annotation_text="Seuil de significativite 5%")
    fig_noise.update_layout(
        height=350, template="plotly_white",
        title=f"Detectabilite du signal (beta = {synthetic_beta:.3f})",
        xaxis_title="Ecart-type du bruit de marche",
        yaxis_title="p-value",
    )
    st.plotly_chart(fig_noise, width="stretch")
    st.caption(
        "**Interpretation :** Ce graphique montre pourquoi la haute frequence est cruciale. "
        "A basse frequence (mensuelle), le bruit agrege est plus eleve, "
        "rendant le signal climatique indetectable."
    )

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
