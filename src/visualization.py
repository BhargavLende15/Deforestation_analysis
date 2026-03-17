from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_total_trend(df: pd.DataFrame):
    """
    Line chart: total forest loss per year (all countries combined).
    """
    yearly = df.groupby("year", as_index=False)["forest_loss_area"].sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(yearly["year"], yearly["forest_loss_area"], marker="o")
    ax.set_title("Total Forest Loss Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Forest loss area")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_country_comparison(df: pd.DataFrame, top_n: int = 10):
    """
    Bar chart: total forest loss per country (top N).
    """
    totals = (
        df.groupby("country", as_index=False)["forest_loss_area"]
        .sum()
        .sort_values("forest_loss_area", ascending=False)
        .head(top_n)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(totals["country"], totals["forest_loss_area"])
    ax.set_title(f"Country-wise Forest Loss (Top {top_n})")
    ax.set_xlabel("Country")
    ax.set_ylabel("Total forest loss area")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_country_trend(df: pd.DataFrame, country: str):
    """
    Line chart: forest loss per year for a selected country.
    """
    cdf = df[df["country"] == country].groupby("year", as_index=False)["forest_loss_area"].sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cdf["year"], cdf["forest_loss_area"], marker="o")
    ax.set_title(f"{country} – Forest Loss Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Forest loss area")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_predictions(pred_df: pd.DataFrame, country: str):
    """
    Line chart: predicted forest loss for future years.
    """
    cdf = pred_df[pred_df["country"] == country]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cdf["year"], cdf["predicted_forest_loss_area"], marker="o", color="orange")
    ax.set_title(f"{country} – Predicted Forest Loss")
    ax.set_xlabel("Year")
    ax.set_ylabel("Predicted forest loss area")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

