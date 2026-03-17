from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = ("year", "country", "forest_loss_area")


def load_dataset(csv_file) -> pd.DataFrame:
    """
    Load the dataset from a file path or a file-like object (Streamlit upload).
    """
    df = pd.read_csv(csv_file)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - ensure required columns exist
    - coerce types
    - drop rows with missing required values
    - standardize country names (strip whitespace)
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Required: {list(REQUIRED_COLUMNS)}"
        )

    out = df.copy()
    out["country"] = out["country"].astype(str).str.strip()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["forest_loss_area"] = pd.to_numeric(out["forest_loss_area"], errors="coerce")

    out = out.dropna(subset=["country", "year", "forest_loss_area"]).copy()
    out["year"] = out["year"].astype(int)
    return out


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Country-wise summary stats table.
    """
    return (
        df.groupby("country", as_index=False)["forest_loss_area"]
        .agg(["count", "sum", "mean", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "n_years",
                "sum": "total_forest_loss_area",
                "mean": "avg_forest_loss_area",
                "min": "min_forest_loss_area",
                "max": "max_forest_loss_area",
            }
        )
    )


def country_year_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot for quick comparison: rows=year, columns=country, values=forest_loss_area.
    """
    return df.pivot_table(
        index="year", columns="country", values="forest_loss_area", aggfunc="sum"
    ).sort_index()

