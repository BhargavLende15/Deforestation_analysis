from __future__ import annotations

import html
import os
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_processing import clean_dataset, load_dataset, summary_statistics
from src.model import get_test_set_actual_predicted, predict_next_years, train_model_per_country
from src.plotly_figures import (
    figure_choropleth_country_loss,
    figure_correlation_heatmap,
    figure_country_bars,
    figure_country_trend,
    figure_country_year_heatmap,
    figure_cumulative,
    figure_model_evaluation,
    figure_multi_country_trend,
    figure_predictions,
    figure_stacked_area_percent,
    figure_total_trend,
    figure_yoy_growth,
)

_PLOTLY_CONFIG = dict(displayModeBar=True, displaylogo=False, modeBarButtonsToRemove=["lasso2d", "select2d"])


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
            .main .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2.75rem;
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
                margin-bottom: 1.5rem;
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
            /* Native Streamlit subheaders used as section titles */
            h3.section-title {
                font-size: 1.35rem;
                font-weight: 700;
                color: #a5d6a7 !important;
                margin: 0 0 0.65rem 0 !important;
                padding-bottom: 0.35rem;
                border-bottom: 2px solid #2e7d32;
            }
            .card {
                background: #1a222d;
                border: 1px solid #2a3544;
                border-radius: 12px;
                padding: 1.1rem 1.2rem 1.2rem 1.2rem;
                margin-bottom: 1.25rem;
                box-shadow: 0 8px 24px rgba(0,0,0,0.35);
            }
            /* Metric tiles (main panel only; avoid wrapping st.metric in raw HTML) */
            section[data-testid="stMain"] [data-testid="stMetric"],
.main [data-testid="stMetric"] {
    background: #161c26;
    border: 1px solid #2a3544;
    border-radius: 12px;
    padding: 0.6rem 0.6rem;
    box-shadow: 0 6px 20px rgba(0,0,0,0.35);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    min-width: 0 !important;
}

/* Fix metric text overflow */
section[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 1.1rem !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

section[data-testid="stMetric"] div[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
}
            section[data-testid="stMain"] [data-testid="stMetric"]:hover,
            .main [data-testid="stMetric"]:hover {
                transform: scale(1.03);
                box-shadow: 0 12px 32px rgba(46, 125, 50, 0.25);
                border-color: #3d6b42;
            }
            hr {
                border: none;
                border-top: 1px solid #2a3544;
                margin: 1.75rem 0 1.5rem 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _card_open() -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)


def _card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def _section_title(title: str) -> None:
    """Section heading: single HTML block to avoid phantom empty inputs below headers."""
    st.markdown(
        f'<h3 class="section-title">{html.escape(title)}</h3>',
        unsafe_allow_html=True,
    )


def _plotly(fig) -> None:
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CONFIG)


st.set_page_config(
    page_title="Deforestation Dashboard",
    layout="wide",
)
_inject_dashboard_css()

st.markdown(
    """
    <div class="dashboard-hero">
        <h1>Deforestation Analysis & Monitoring</h1>
        <p>Upload forest loss data, explore trends on an interactive map, and review simple linear projections.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Data & controls")
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("CSV (`year`, `country`, `forest_loss_area`)", type=["csv"])
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    horizon = st.slider("Prediction horizon (years ahead)", min_value=5, max_value=10, value=5)
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    top_n = st.slider("Top N countries (bar chart)", min_value=3, max_value=25, value=6)
    st.caption("Charts support hover tooltips (country, year, values). Map uses country names or 3-letter ISO codes.")

if uploaded is None:
    st.info("Upload a CSV with columns: **year**, **country**, **forest_loss_area**")
    st.stop()

raw_df = load_dataset(uploaded)

try:
    df = clean_dataset(raw_df)
except Exception as e:
    st.error(f"Data validation/cleaning failed: {e}")
    st.stop()

st.markdown("---")
_section_title("Cleaned data & summary")
_card_open()
c1, c2 = st.columns([1, 1])
with c1:
    st.markdown("**Preview** (first 50 rows)")
    st.dataframe(df.head(50), use_container_width=True)
with c2:
    st.markdown("**Dataset health**")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Rows", f"{len(df):,}")
    with m2:
        st.metric("Countries", f"{df['country'].nunique():,}")
    with m3:
        st.metric("Min year", f"{int(df['year'].min())}")
    with m4:
        st.metric("Max year", f"{int(df['year'].max())}")
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("**Country-wise statistics**")
    st.dataframe(summary_statistics(df), use_container_width=True)
_card_close()

st.markdown("---")
_section_title("Exploratory analysis")
_card_open()
left, right = st.columns([1, 1])
with left:
    _plotly(figure_total_trend(df))
with right:
    _plotly(figure_country_bars(df, top_n=top_n))
_card_close()

st.markdown("---")
_section_title("Correlation analysis")
_card_open()
st.caption(
    "Pearson correlations across **numeric** columns in your file (e.g. **year**, **forest_loss_area**, "
    "**forest_loss** if present). Constant columns are omitted."
)
_plotly(figure_correlation_heatmap(df))
_card_close()

st.markdown("---")
_section_title("Global forest loss map")
_card_open()
st.caption(
    "Total **forest_loss_area** by **country**. Use **ISO 3166-1 alpha-3** codes (e.g. IND, USA) "
    "for best geometry match; otherwise English **country names** are used."
)
_plotly(figure_choropleth_country_loss(df))
_card_close()

st.markdown("---")
_section_title("Advanced analysis")
_card_open()
r1c1, r1c2 = st.columns([1, 1])
with r1c1:
    _plotly(figure_multi_country_trend(df, max_countries=8))
with r1c2:
    _plotly(figure_stacked_area_percent(df, max_countries=10))
r2c1, r2c2 = st.columns([1, 1])
with r2c1:
    _plotly(figure_yoy_growth(df, max_countries=10))
with r2c2:
    _plotly(figure_cumulative(df, max_countries=10))
_plotly(figure_country_year_heatmap(df))
_card_close()

st.markdown("---")
_section_title("Country deep dive")
_card_open()
country = st.selectbox(
    "Select a country for detailed trend, model, and evaluation",
    options=sorted(df["country"].unique().tolist()),
)
_plotly(figure_country_trend(df, country))
_card_close()

st.markdown("---")
_section_title("Machine learning (linear regression)")
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
    _plotly(figure_predictions(pred_df, country))

    eval_df = get_test_set_actual_predicted(df, country)
    _plotly(figure_model_evaluation(eval_df, country))

    with st.expander("See prediction table"):
        st.dataframe(
            pred_df[pred_df["country"] == country],
            use_container_width=True,
        )
    _card_close()

st.markdown("---")
_section_title("Export")
_card_open()
processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(processed_dir, exist_ok=True)
export_name = st.text_input("Export filename (CSV)", value="cleaned_deforestation_data.csv")
export_path = os.path.join(processed_dir, export_name)
if st.button("Export cleaned CSV to `data/processed/`"):
    df.to_csv(export_path, index=False)
    st.success(f"Saved: {export_path}")
_card_close()
