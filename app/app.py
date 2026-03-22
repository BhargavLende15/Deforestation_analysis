from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_processing import clean_dataset, load_dataset, summary_statistics
from src.model import get_test_set_actual_predicted, predict_next_years, train_model_per_country
from src.visualization import (
    plot_country_comparison,
    plot_country_trend,
    plot_cumulative,
    plot_heatmap,
    plot_model_evaluation,
    plot_multi_country_trend,
    plot_predictions,
    plot_stacked_area,
    plot_total_trend,
    plot_yoy_growth,
)


def _inject_dashboard_css() -> None:
    st.markdown(
        """
        <style>
            html, body, [class*="css"] {
                font-family: 'Segoe UI', system-ui, sans-serif;
            }
            .stApp {
                background: linear-gradient(165deg, #0f1419 0%, #151b22 45%, #0d1218 100%);
                color: #e8eaed;
            }
            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #12181f 0%, #0c1016 100%);
                border-right: 1px solid #1f2a35;
            }
            section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label {
                color: #c5d0dc !important;
            }
            .dashboard-hero {
                padding: 0.25rem 0 1rem 0;
                border-bottom: 1px solid #243040;
                margin-bottom: 1.25rem;
            }
            .dashboard-hero h1 {
                color: #e8f5e9;
                font-size: 2rem;
                font-weight: 700;
                letter-spacing: -0.02em;
                margin-bottom: 0.35rem;
            }
            .dashboard-hero p {
                color: #8fa3b0;
                font-size: 1rem;
                margin: 0;
            }
            .section-header {
                font-size: 1.35rem;
                font-weight: 700;
                color: #a5d6a7;
                margin: 1.5rem 0 0.75rem 0;
                padding-bottom: 0.35rem;
                border-bottom: 2px solid #2e7d32;
                display: inline-block;
                width: 100%;
            }
            .card {
                background: #1a222d;
                border: 1px solid #2a3544;
                border-radius: 12px;
                padding: 1rem 1.15rem 1.1rem 1.15rem;
                margin-bottom: 1rem;
                box-shadow: 0 8px 24px rgba(0,0,0,0.35);
            }
            div[data-testid="stMetric"] {
                background: #1a222d;
                border: 1px solid #2a3544;
                border-radius: 10px;
                padding: 0.5rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _card_open() -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)


def _card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def _section_header(title: str) -> None:
    st.markdown(f'<p class="section-header">{title}</p>', unsafe_allow_html=True)


def _per_country_trend_slope(df: pd.DataFrame, country: str) -> float:
    s = df[df["country"] == country].groupby("year")["forest_loss_area"].sum().sort_index()
    if len(s) < 2:
        return float("nan")
    x = s.index.to_numpy(dtype=float)
    y = s.to_numpy(dtype=float)
    m, _ = np.polyfit(x, y, 1)
    return float(m)


def compute_key_observations(df: pd.DataFrame) -> list[str]:
    """Bullet-style observations for the dashboard."""
    bullets: list[str] = []
    if df.empty:
        return ["- No rows available after cleaning."]

    by_country = df.groupby("country", as_index=False)["forest_loss_area"].sum()
    top_country = by_country.loc[by_country["forest_loss_area"].idxmax(), "country"]
    bullets.append(
        f"- **Highest cumulative loss:** **{top_country}** "
        f"({by_country['forest_loss_area'].max():,.2f} total forest loss area in the dataset)."
    )

    slopes = []
    for c in df["country"].unique():
        m = _per_country_trend_slope(df, c)
        if np.isfinite(m):
            slopes.append((c, m))
    if slopes:
        fastest = max(slopes, key=lambda t: t[1])
        bullets.append(
            f"- **Steepest upward trend (linear slope vs year):** **{fastest[0]}** "
            f"(approx. **{fastest[1]:+.4f}** loss units per year on average)."
        )

    yearly = df.groupby("year", as_index=False)["forest_loss_area"].sum().sort_values("year")
    if len(yearly) >= 2:
        x = yearly["year"].to_numpy(dtype=float)
        y = yearly["forest_loss_area"].to_numpy(dtype=float)
        glob_slope, _ = np.polyfit(x, y, 1)
        direction = "increasing" if glob_slope > 0 else "decreasing" if glob_slope < 0 else "flat"
        bullets.append(
            f"- **Overall global trend:** forest loss appears **{direction}** over time "
            f"(aggregate slope ≈ **{glob_slope:+.4f}** per year)."
        )

    peak_row = yearly.loc[yearly["forest_loss_area"].idxmax()]
    bullets.append(
        f"- **Peak global loss year:** **{int(peak_row['year'])}** "
        f"with **{peak_row['forest_loss_area']:,.2f}** total area."
    )

    if len(yearly) >= 3:
        yoy = yearly.set_index("year")["forest_loss_area"].pct_change().dropna() * 100.0
        if not yoy.empty:
            z = (yoy - yoy.mean()) / (yoy.std() + 1e-12)
            spikes = yoy[np.abs(z) > 1.5]
            if not spikes.empty:
                top_spike_year = int(spikes.abs().idxmax())
                top_spike_val = float(spikes.loc[top_spike_year])
                bullets.append(
                    f"- **Notable volatility:** **{top_spike_year}** saw an exceptional "
                    f"year-over-year change of **{top_spike_val:+.1f}%** vs the prior year "
                    f"(relative to typical fluctuations)."
                )
            else:
                bullets.append(
                    "- **Volatility:** no extreme year-over-year spikes detected "
                    "beyond ~1.5 standard deviations from the mean."
                )

    return bullets


def compute_conclusion_bullets(
    df: pd.DataFrame,
    models: dict,
    pred_df: pd.DataFrame | None,
    selected_country: str,
) -> list[str]:
    out: list[str] = []
    yearly = df.groupby("year", as_index=False)["forest_loss_area"].sum().sort_values("year")
    if len(yearly) >= 2:
        x = yearly["year"].to_numpy(dtype=float)
        y = yearly["forest_loss_area"].to_numpy(dtype=float)
        glob_slope, _ = np.polyfit(x, y, 1)
        if glob_slope > 0:
            out.append(
                "- **Deforestation signal:** combined national losses in this file **trend upward** "
                "over the sampled years, which is consistent with sustained pressure on forest cover."
            )
        elif glob_slope < 0:
            out.append(
                "- **Deforestation signal:** aggregate losses **trend downward** in this dataset—"
                "interpret with care (policy, data coverage, or methodology may drive this)."
            )
        else:
            out.append("- **Deforestation signal:** aggregate losses look **roughly flat** across years in this file.")

    top3 = (
        df.groupby("country")["forest_loss_area"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )
    if top3:
        out.append(
            "- **Most affected (by total recorded loss):** "
            + ", ".join(f"**{c}**" for c in top3)
            + "."
        )

    if models:
        r2_vals = [res.r2 for res in models.values() if np.isfinite(res.r2)]
        if r2_vals:
            med_r2 = float(np.nanmedian(r2_vals))
            out.append(
                f"- **Linear trend model:** across countries with enough history, "
                f"typical hold-out **R² ≈ {med_r2:.2f}**—a simple `year → loss` line is "
                f"{'moderately' if med_r2 > 0.4 else 'weakly'} explanatory here."
            )
    else:
        out.append(
            "- **Linear trend model:** not enough per-country history in this file to train "
            "the default regression for most countries."
        )

    if pred_df is not None and selected_country in pred_df["country"].values:
        sub = pred_df[pred_df["country"] == selected_country].sort_values("year")
        if len(sub) >= 2:
            first, last = sub["predicted_forest_loss_area"].iloc[0], sub["predicted_forest_loss_area"].iloc[-1]
            proj = "rising" if last > first else "falling" if last < first else "stable"
            out.append(
                f"- **Projection for {selected_country}:** the fitted line suggests **{proj}** "
                f"predicted loss over the next horizon shown in the chart."
            )

    out.append(
        "- **Real-world implication:** persistent or rising forest loss reduces carbon storage "
        "and biodiversity; monitoring these trends helps target conservation and restoration."
    )
    return out


st.set_page_config(page_title="Deforestation Dashboard", layout="wide", )
_inject_dashboard_css()

st.markdown(
    """
    <div class="dashboard-hero">
        <h1>Deforestation Analysis & Monitoring</h1>
        <p>Upload forest loss data, explore trends, and review simple linear projections.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Data & controls")
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("CSV upload (`year`, `country`, `forest_loss_area`)", type=["csv"])
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    horizon = st.slider("Prediction horizon (years ahead)", min_value=5, max_value=10, value=5)
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    top_n = st.slider("Top N countries (bar chart)", min_value=3, max_value=25, value=6)
    st.caption("Tip: use a wide layout and scroll for advanced charts and insights.")

if uploaded is None:
    st.info("Upload a CSV with columns: **year**, **country**, **forest_loss_area**")
    st.stop()

raw_df = load_dataset(uploaded)

try:
    df = clean_dataset(raw_df)
except Exception as e:
    st.error(f"Data validation/cleaning failed: {e}")
    st.stop()

_card_open()
_section_header("Cleaned data & summary")
c1, c2 = st.columns([1, 1])
with c1:
    st.markdown("**Preview** (first 50 rows)")
    st.dataframe(df.head(50), use_container_width=True)
with c2:
    st.markdown("**Dataset health**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{len(df):,}")
    m2.metric("Countries", f"{df['country'].nunique():,}")
    m3.metric("Min year", f"{int(df['year'].min())}")
    m4.metric("Max year", f"{int(df['year'].max())}")
    st.markdown("**Country-wise statistics**")
    st.dataframe(summary_statistics(df), use_container_width=True)
_card_close()

_section_header("Exploratory analysis")
_card_open()
left, right = st.columns([1, 1])
with left:
    st.pyplot(plot_total_trend(df), clear_figure=True)
with right:
    st.pyplot(plot_country_comparison(df, top_n=top_n), clear_figure=True)
_card_close()

_section_header("Advanced analysis")
_card_open()
r1c1, r1c2 = st.columns([1, 1])
with r1c1:
    st.pyplot(plot_multi_country_trend(df), clear_figure=True)
with r1c2:
    st.pyplot(plot_stacked_area(df), clear_figure=True)
r2c1, r2c2 = st.columns([1, 1])
with r2c1:
    st.pyplot(plot_yoy_growth(df), clear_figure=True)
with r2c2:
    st.pyplot(plot_cumulative(df), clear_figure=True)
st.pyplot(plot_heatmap(df), clear_figure=True)
_card_close()

_card_open()
_section_header("Country deep dive")
country = st.selectbox(
    "Select a country for detailed trend, model, and evaluation",
    options=sorted(df["country"].unique().tolist()),
)
st.pyplot(plot_country_trend(df, country), clear_figure=True)
_card_close()

_section_header("Machine learning (linear regression)")
models = train_model_per_country(df)
pred_df: pd.DataFrame | None = None

if not models:
    st.warning("No country has enough distinct years (≥3) to train a linear regression model.")
elif country not in models:
    st.warning(
        f"**{country}** does not have enough distinct years to train the model. "
        "Pick another country or enrich the time series."
    )
else:
    res = models[country]
    _card_open()
    st.write(
        {
            "model": "LinearRegression (`forest_loss_area` ~ `year`)",
            "R² (hold-out)": res.r2,
            "MAE (hold-out)": res.mae,
        }
    )
    pred_df = predict_next_years(models, last_year=int(df["year"].max()), n_years=horizon)
    st.pyplot(plot_predictions(pred_df, country), clear_figure=True)

    eval_df = get_test_set_actual_predicted(df, country)
    st.pyplot(plot_model_evaluation(eval_df, country), clear_figure=True)

    with st.expander("See prediction table"):
        st.dataframe(
            pred_df[pred_df["country"] == country],
            use_container_width=True,
        )
    _card_close()

if models and pred_df is None:
    pred_df = predict_next_years(models, last_year=int(df["year"].max()), n_years=horizon)

_section_header("Insights & conclusion")
_card_open()
st.markdown("### Key observations")
obs = compute_key_observations(df)
st.markdown("\n".join(obs))
st.markdown("### Conclusion")
concl = compute_conclusion_bullets(df, models, pred_df, country)
st.markdown("\n".join(concl))
_card_close()

_section_header("Export")
_card_open()
processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(processed_dir, exist_ok=True)
export_name = st.text_input("Export filename (CSV)", value="cleaned_deforestation_data.csv")
export_path = os.path.join(processed_dir, export_name)
if st.button("Export cleaned CSV to `data/processed/`"):
    df.to_csv(export_path, index=False)
    st.success(f"Saved: {export_path}")
_card_close()
