from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_processing import clean_dataset, load_dataset, summary_statistics
from src.model import predict_next_years, train_model_per_country
from src.visualization import (
    plot_country_comparison,
    plot_country_trend,
    plot_predictions,
    plot_total_trend,
)


st.set_page_config(page_title="Deforestation Dashboard", layout="wide")

st.title("Deforestation Analysis & Monitoring Dashboard")
st.caption("Indian Subcontinent (CSV-based workflow)")

with st.sidebar:
    st.header("Upload data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    horizon = st.slider("Prediction horizon (years)", min_value=5, max_value=10, value=5)
    top_n = st.slider("Top N countries (bar chart)", min_value=3, max_value=25, value=6)


if uploaded is None:
    st.info("Upload a CSV with columns: year, country, forest_loss_area")
    st.stop()

raw_df = load_dataset(uploaded)

try:
    df = clean_dataset(raw_df)
except Exception as e:
    st.error(f"Data validation/cleaning failed: {e}")
    st.stop()

st.subheader("Cleaned dataset preview")
st.dataframe(df.head(50), use_container_width=True)

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Summary statistics (country-wise)")
    st.dataframe(summary_statistics(df), use_container_width=True)
with col2:
    st.subheader("Dataset health")
    st.write(
        {
            "rows": int(len(df)),
            "countries": int(df["country"].nunique()),
            "min_year": int(df["year"].min()),
            "max_year": int(df["year"].max()),
        }
    )

st.divider()
st.subheader("Exploratory analysis")

left, right = st.columns([1, 1])
with left:
    st.pyplot(plot_total_trend(df), clear_figure=True)
with right:
    st.pyplot(plot_country_comparison(df, top_n=top_n), clear_figure=True)

country = st.selectbox(
    "Select a country for detailed trend + prediction",
    options=sorted(df["country"].unique().tolist()),
)

st.pyplot(plot_country_trend(df, country), clear_figure=True)

st.divider()
st.subheader("Machine Learning (Linear Regression)")

models = train_model_per_country(df)
if country not in models:
    st.warning(
        "Not enough data points to train a model for this country. "
        "Add at least 3 distinct years."
    )
    st.stop()

res = models[country]
st.write(
    {
        "model": "LinearRegression (forest_loss_area ~ year)",
        "R2": res.r2,
        "MAE": res.mae,
    }
)

pred_df = predict_next_years(models, last_year=int(df["year"].max()), n_years=horizon)
st.pyplot(plot_predictions(pred_df, country), clear_figure=True)

with st.expander("See prediction table"):
    st.dataframe(
        pred_df[pred_df["country"] == country],
        use_container_width=True,
    )

st.divider()
st.subheader("Export")

processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(processed_dir, exist_ok=True)

export_name = st.text_input("Export filename (CSV)", value="cleaned_deforestation_data.csv")
export_path = os.path.join(processed_dir, export_name)

if st.button("Export cleaned CSV to data/processed/"):
    df.to_csv(export_path, index=False)
    st.success(f"Saved: {export_path}")

