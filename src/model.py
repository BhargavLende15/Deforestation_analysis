from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


@dataclass
class ModelResult:
    model: LinearRegression
    r2: float
    mae: float


def _prepare_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df[["year"]].to_numpy(dtype=float)
    y = df["forest_loss_area"].to_numpy(dtype=float)
    return X, y


def train_model_per_country(df: pd.DataFrame) -> Dict[str, ModelResult]:
    """
    Train a separate Linear Regression model per country:
        forest_loss_area ~ year
    """
    results: Dict[str, ModelResult] = {}
    for country, cdf in df.groupby("country"):
        cdf = cdf.sort_values("year")
        if cdf["year"].nunique() < 3:
            # Not enough points to train a meaningful trend model
            continue

        X, y = _prepare_xy(cdf)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[country] = ModelResult(
            model=model,
            r2=float(r2_score(y_test, y_pred)) if len(y_test) > 1 else float("nan"),
            mae=float(mean_absolute_error(y_test, y_pred)),
        )

    return results


def predict_next_years(
    models: Dict[str, ModelResult],
    last_year: int,
    n_years: int = 5,
) -> pd.DataFrame:
    """
    Predict the next N years for each country.
    """
    future_years = np.arange(last_year + 1, last_year + n_years + 1, dtype=int)
    rows = []
    for country, res in models.items():
        preds = res.model.predict(future_years.reshape(-1, 1))
        for year, pred in zip(future_years, preds):
            rows.append(
                {
                    "country": country,
                    "year": int(year),
                    "predicted_forest_loss_area": float(pred),
                }
            )
    return pd.DataFrame(rows).sort_values(["country", "year"])


def get_test_set_actual_predicted(df: pd.DataFrame, country: str) -> Optional[pd.DataFrame]:
    """
    Fit the same per-country linear model as train_model_per_country and return
    hold-out test predictions for actual vs predicted plots.
    """
    cdf = df[df["country"] == country].sort_values("year")
    if cdf["year"].nunique() < 3:
        return None

    X, y = _prepare_xy(cdf)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return pd.DataFrame(
        {
            "actual": y_test.ravel(),
            "predicted": y_pred.ravel(),
        }
    )

