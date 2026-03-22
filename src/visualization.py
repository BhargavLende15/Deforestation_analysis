from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Consistent chart styling (readable on light figure background inside dark UI cards)
FIGSIZE_WIDE = (10, 5)
FIGSIZE_TALL = (10, 6)


def _apply_base_style(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.14, linestyle=":", linewidth=0.7)
    ax.tick_params(axis="both", labelsize=10)


def plot_total_trend(df: pd.DataFrame):
    """
    Line chart: total forest loss per year (all countries combined).
    """
    yearly = df.groupby("year", as_index=False)["forest_loss_area"].sum()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(yearly["year"], yearly["forest_loss_area"], marker="o", color="#2e7d32", linewidth=2)
    ax.set_title("Total Forest Loss Over Time", fontsize=14, fontweight="bold", color="#1b5e20")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Forest loss area", fontsize=11)
    _apply_base_style(ax)
    fig.tight_layout()
    return fig


def plot_country_comparison(df: pd.DataFrame, top_n: int = 10):
    """
    Horizontal bar chart: total forest loss per country (top N), sorted descending.
    """
    totals = (
        df.groupby("country", as_index=False)["forest_loss_area"]
        .sum()
        .sort_values("forest_loss_area", ascending=True)
        .tail(top_n)
    )
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    colors = plt.cm.Greens(np.linspace(0.35, 0.85, len(totals)))
    ax.barh(totals["country"], totals["forest_loss_area"], color=colors, edgecolor="#1b5e20", linewidth=0.5)
    ax.set_title(f"Country-wise Forest Loss (Top {top_n})", fontsize=14, fontweight="bold", color="#1b5e20")
    ax.set_xlabel("Total forest loss area", fontsize=11)
    ax.set_ylabel("Country", fontsize=11)
    _apply_base_style(ax)
    fig.tight_layout()
    return fig


def plot_country_trend(df: pd.DataFrame, country: str):
    """
    Line chart: forest loss per year for a selected country.
    """
    cdf = df[df["country"] == country].groupby("year", as_index=False)["forest_loss_area"].sum()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(cdf["year"], cdf["forest_loss_area"], marker="o", color="#388e3c", linewidth=2)
    ax.set_title(f"{country} – Forest Loss Over Time", fontsize=14, fontweight="bold", color="#1b5e20")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Forest loss area", fontsize=11)
    _apply_base_style(ax)
    fig.tight_layout()
    return fig


def plot_predictions(pred_df: pd.DataFrame, country: str):
    """
    Line chart: predicted forest loss for future years.
    """
    cdf = pred_df[pred_df["country"] == country]
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(
        cdf["year"],
        cdf["predicted_forest_loss_area"],
        marker="o",
        color="#f57c00",
        linewidth=2,
    )
    ax.set_title(f"{country} – Predicted Forest Loss", fontsize=14, fontweight="bold", color="#e65100")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Predicted forest loss area", fontsize=11)
    _apply_base_style(ax)
    fig.tight_layout()
    return fig


def plot_multi_country_trend(df: pd.DataFrame, max_countries: int = 18):
    """
    Forest loss vs year with one line per country (capped for readability).
    """
    wide = df.pivot_table(
        index="year",
        columns="country",
        values="forest_loss_area",
        aggfunc="sum",
    ).sort_index()
    totals = wide.sum(axis=0).sort_values(ascending=False)
    countries = totals.head(max_countries).index.tolist()
    plot_df = wide[countries]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for col in plot_df.columns:
        s = plot_df[col].dropna()
        if len(s) == 0:
            continue
        ax.plot(s.index, s.values, marker=".", linewidth=1.5, alpha=0.85, label=col)
    ax.set_title(
        "Multi-country trend — forest loss over time",
        fontsize=14,
        fontweight="bold",
        color="#1b5e20",
    )
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Forest loss area", fontsize=11)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, framealpha=0.9, ncol=1)
    _apply_base_style(ax)
    fig.tight_layout()
    return fig


def plot_stacked_area(df: pd.DataFrame, max_countries: int = 12):
    """
    Stacked area chart: each country's share of annual loss (normalized to 100% per year).
    Countries beyond max_countries are aggregated into "Other" so totals reach 100%.
    """
    wide = df.pivot_table(
        index="year",
        columns="country",
        values="forest_loss_area",
        aggfunc="sum",
    ).sort_index()
    wide = wide.fillna(0)
    totals = wide.sum(axis=0).sort_values(ascending=False)
    top = totals.head(max_countries).index.tolist()
    rest = [c for c in wide.columns if c not in top]
    sub = wide[top].copy()
    if rest:
        sub["Other"] = wide[rest].sum(axis=1)
    row_sum = sub.sum(axis=1)
    row_sum = row_sum.replace(0, np.nan)
    pct = sub.div(row_sum, axis=0) * 100.0
    pct = pct.fillna(0)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    years = pct.index.to_numpy()
    ax.stackplot(
        years,
        *[pct[c].to_numpy() for c in pct.columns],
        labels=pct.columns,
        alpha=0.85,
    )
    ax.set_title(
        "Share of annual forest loss by country (100% per year)",
        fontsize=14,
        fontweight="bold",
        color="#1b5e20",
    )
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Share of global annual loss (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, framealpha=0.9)
    _apply_base_style(ax)
    fig.tight_layout()
    return fig


def plot_yoy_growth(df: pd.DataFrame, max_countries: int = 15):
    """
    Year-over-year % change in forest loss per country (line chart).
    """
    parts = []
    for country, g in df.groupby("country"):
        s = g.groupby("year", as_index=False)["forest_loss_area"].sum().sort_values("year")
        if len(s) < 2:
            continue
        s = s.set_index("year")["forest_loss_area"]
        yoy = s.pct_change() * 100.0
        yoy = yoy.dropna()
        if yoy.empty:
            continue
        parts.append(pd.DataFrame({"year": yoy.index, "country": country, "yoy_pct": yoy.values}))
    if not parts:
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        ax.text(0.5, 0.5, "Not enough years per country for YoY growth.", ha="center", va="center")
        ax.axis("off")
        return fig

    long = pd.concat(parts, ignore_index=True)
    top = long.groupby("country")["yoy_pct"].apply(lambda x: np.nanmean(np.abs(x))).sort_values(ascending=False)
    keep = top.head(max_countries).index
    long = long[long["country"].isin(keep)]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for country, g in long.groupby("country"):
        g = g.sort_values("year")
        ax.plot(g["year"], g["yoy_pct"], marker=".", linewidth=1.2, alpha=0.85, label=country)
    ax.axhline(0, color="#666", linestyle="--", linewidth=1)
    ax.set_title(
        "Year-over-year change in forest loss (%)",
        fontsize=14,
        fontweight="bold",
        color="#1b5e20",
    )
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("YoY change (%)", fontsize=11)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, framealpha=0.9)
    _apply_base_style(ax)
    fig.tight_layout()
    return fig


def plot_heatmap(df: pd.DataFrame):
    """
    Heatmap: rows = country, columns = year, values = forest_loss_area.
    """
    mat = df.pivot_table(
        index="country",
        columns="year",
        values="forest_loss_area",
        aggfunc="sum",
    )
    mat = mat.sort_index()

    fig_h = max(6, min(24, 0.35 * len(mat) + 4))
    fig, ax = plt.subplots(figsize=(FIGSIZE_WIDE[0], fig_h))
    sns.heatmap(
        mat,
        ax=ax,
        cmap="Greens",
        linewidths=0.3,
        linecolor="#e0e0e0",
        cbar_kws={"label": "Forest loss area"},
    )
    ax.set_title(
        "Forest loss heatmap (country × year)",
        fontsize=14,
        fontweight="bold",
        color="#1b5e20",
    )
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Country", fontsize=11)
    fig.tight_layout()
    return fig


def plot_cumulative(df: pd.DataFrame, max_countries: int = 15):
    """
    Running total of forest loss per country over time.
    """
    rows = []
    for country, g in df.groupby("country"):
        s = g.groupby("year", as_index=False)["forest_loss_area"].sum().sort_values("year")
        s["cumulative"] = s["forest_loss_area"].cumsum()
        s["country"] = country
        rows.append(s[["year", "country", "cumulative"]])
    if not rows:
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        ax.text(0.5, 0.5, "No data for cumulative plot.", ha="center", va="center")
        ax.axis("off")
        return fig

    long = pd.concat(rows, ignore_index=True)
    totals = long.groupby("country")["cumulative"].max().sort_values(ascending=False)
    keep = totals.head(max_countries).index
    long = long[long["country"].isin(keep)]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for country, g in long.groupby("country"):
        g = g.sort_values("year")
        ax.plot(g["year"], g["cumulative"], marker=".", linewidth=1.5, alpha=0.9, label=country)
    ax.set_title(
        "Cumulative forest loss over time",
        fontsize=14,
        fontweight="bold",
        color="#1b5e20",
    )
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Cumulative forest loss area", fontsize=11)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, framealpha=0.9)
    _apply_base_style(ax)
    fig.tight_layout()
    return fig


def plot_model_evaluation(eval_df: Optional[pd.DataFrame], country: str):
    """
    Scatter: actual vs predicted on the hold-out test set; diagonal reference line.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    if eval_df is None or eval_df.empty:
        ax.text(
            0.5,
            0.5,
            "Not enough data to evaluate the model for this country.",
            ha="center",
            va="center",
        )
        ax.axis("off")
        fig.tight_layout()
        return fig

    x = eval_df["actual"].to_numpy()
    y = eval_df["predicted"].to_numpy()
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    if lo == hi:
        hi = lo + 1e-9
    ax.scatter(x, y, alpha=0.85, color="#2e7d32", edgecolors="#1b5e20", s=55)
    ax.plot([lo, hi], [lo, hi], color="#f57c00", linestyle="--", linewidth=2, label="Perfect fit (y = x)")
    ax.set_title(
        f"{country} — actual vs predicted (test set)",
        fontsize=14,
        fontweight="bold",
        color="#1b5e20",
    )
    ax.set_xlabel("Actual forest loss area", fontsize=11)
    ax.set_ylabel("Predicted forest loss area", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    _apply_base_style(ax)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig
