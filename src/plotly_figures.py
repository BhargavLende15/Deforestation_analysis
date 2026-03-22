from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# Consistent dark styling (matches dashboard CSS)
_LAYOUT_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(26, 34, 45, 0.92)",
    plot_bgcolor="rgba(18, 24, 32, 0.95)",
    font=dict(color="#e8eaed", family="Segoe UI, sans-serif", size=12),
    margin=dict(l=54, r=28, t=56, b=48),
    hoverlabel=dict(bgcolor="#1a222d", font=dict(size=13)),
)

_COLOR_SEQ = px.colors.sequential.Greens


def _apply_chart_axes(fig: go.Figure, *, show_x_grid: bool = False, show_y_grid: bool = True) -> None:
    fig.update_xaxes(
        showgrid=show_x_grid,
        zeroline=False,
        gridcolor="rgba(100, 120, 110, 0.2)",
    )
    fig.update_yaxes(
        showgrid=show_y_grid,
        zeroline=False,
        gridcolor="rgba(100, 120, 110, 0.18)",
    )


def _title_layout(title: str) -> dict:
    d = {**_LAYOUT_BASE}
    d["title"] = dict(text=title, font=dict(size=16, color="#a5d6a7"))
    return d


def numeric_columns_for_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric features for correlation (row-level). Drops constant columns."""
    num = df.select_dtypes(include=[np.number]).copy()
    if "forest_loss" in df.columns and "forest_loss" not in num.columns:
        num["forest_loss"] = pd.to_numeric(df["forest_loss"], errors="coerce")
    keep = [c for c in num.columns if num[c].nunique(dropna=True) > 1]
    return num[keep]


def figure_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation matrix of numeric columns (year, forest_loss_area, forest_loss, etc.)."""
    num = numeric_columns_for_correlation(df)
    if num.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least two varying numeric columns for a correlation matrix.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#8fa3b0"),
        )
        fig.update_layout(**_title_layout("Correlation matrix"), xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    corr = num.corr(numeric_only=True)
    labels = [c.replace("_", " ").title() for c in corr.columns]
    fig = px.imshow(
        corr.values,
        x=labels,
        y=labels,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        aspect="equal",
        labels=dict(color="Pearson r"),
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
    )
    fig.update_layout(**_title_layout("Correlation analysis (numeric features)"))
    fig.update_xaxes(side="bottom", tickangle=-30)
    _apply_chart_axes(fig, show_x_grid=False, show_y_grid=False)
    return fig


def figure_choropleth_country_loss(df: pd.DataFrame) -> go.Figure:
    """Total forest loss by country on a world map."""
    agg = df.groupby("country", as_index=False)["forest_loss_area"].sum()
    if agg.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No country aggregates to display on the map.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#8fa3b0"),
        )
        fig.update_layout(**_title_layout("Global forest loss map"), geo=dict(showframe=False))
        return fig

    iso_like = agg["country"].astype(str).str.fullmatch(r"[A-Za-z]{3}", na=False)
    if bool(iso_like.all()) and len(agg):
        agg = agg.assign(_map_loc=agg["country"].str.upper())
        locmode = "ISO-3"
    else:
        agg = agg.assign(_map_loc=agg["country"])
        locmode = "country names"

    fig = px.choropleth(
        agg,
        locations="_map_loc",
        locationmode=locmode,
        color="forest_loss_area",
        color_continuous_scale="Greens",
        labels={"forest_loss_area": "Forest loss (total)"},
    )
    fig.update_traces(
        hovertemplate="<b>%{location}</b><br>Total forest loss: %{z:,.2f}<extra></extra>",
        marker_line_width=0.5,
        marker_line_color="rgba(20,30,25,0.6)",
    )
    fig.update_layout(
        **_title_layout("Global forest loss map"),
        geo=dict(
            bgcolor="rgba(18,24,32,0.9)",
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth",
        ),
        coloraxis_colorbar=dict(
            title=dict(text="Total loss", side="right"),
            tickformat=",.0f",
        ),
    )
    return fig


def figure_total_trend(df: pd.DataFrame) -> go.Figure:
    yearly = df.groupby("year", as_index=False)["forest_loss_area"].sum()
    fig = px.line(
        yearly,
        x="year",
        y="forest_loss_area",
        markers=True,
        color_discrete_sequence=["#66bb6a"],
    )
    fig.update_traces(
        name="Global total",
        hovertemplate="Year: %{x}<br>Total forest loss: %{y:,.4f}<extra></extra>",
        line=dict(width=2.5),
    )
    fig.update_layout(**_title_layout("Total forest loss over time"), showlegend=False)
    fig.update_yaxes(title_text="Forest loss area")
    fig.update_xaxes(title_text="Year")
    _apply_chart_axes(fig, show_x_grid=False, show_y_grid=True)
    return fig


def figure_country_bars(df: pd.DataFrame, top_n: int) -> go.Figure:
    totals = (
        df.groupby("country", as_index=False)["forest_loss_area"]
        .sum()
        .sort_values("forest_loss_area", ascending=True)
        .tail(top_n)
    )
    fig = px.bar(
        totals,
        x="forest_loss_area",
        y="country",
        orientation="h",
        color="forest_loss_area",
        color_continuous_scale="Greens",
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Total forest loss: %{x:,.4f}<extra></extra>",
        marker_line_width=0,
    )
    fig.update_layout(**_title_layout(f"Country-wise forest loss (top {top_n})"), showlegend=False)
    fig.update_xaxes(title_text="Total forest loss area")
    fig.update_yaxes(title_text="Country")
    _apply_chart_axes(fig, show_x_grid=True, show_y_grid=False)
    return fig


def figure_multi_country_trend(df: pd.DataFrame, max_countries: int = 8) -> go.Figure:
    wide = df.pivot_table(
        index="year",
        columns="country",
        values="forest_loss_area",
        aggfunc="sum",
    ).sort_index()
    totals = wide.sum(axis=0).sort_values(ascending=False)
    countries = totals.head(max_countries).index.tolist()
    long = wide[countries].reset_index().melt(id_vars="year", var_name="country", value_name="forest_loss_area")
    long = long.dropna(subset=["forest_loss_area"])

    fig = px.line(
        long,
        x="year",
        y="forest_loss_area",
        color="country",
        markers=False,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>Year: %{x}<br>Forest loss: %{y:,.4f}<extra></extra>",
        line=dict(width=2),
    )
    fig.update_layout(
        **_title_layout("Multi-country trend"),
        legend=dict(
            title="Country",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=10),
        ),
    )
    fig.update_yaxes(title_text="Forest loss area")
    fig.update_xaxes(title_text="Year")
    _apply_chart_axes(fig, show_x_grid=False, show_y_grid=True)
    return fig


def _stacked_percent_long(df: pd.DataFrame, max_countries: int = 10) -> pd.DataFrame:
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
    row_sum = sub.sum(axis=1).replace(0, np.nan)
    pct = sub.div(row_sum, axis=0) * 100.0
    pct = pct.fillna(0)
    return pct.reset_index().melt(id_vars="year", var_name="country", value_name="share_pct")


def figure_stacked_area_percent(df: pd.DataFrame, max_countries: int = 10) -> go.Figure:
    long = _stacked_percent_long(df, max_countries=max_countries)
    fig = px.area(
        long,
        x="year",
        y="share_pct",
        color="country",
        line_shape="linear",
        color_discrete_sequence=_COLOR_SEQ,
    )
    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>Year: %{x}<br>Share of annual loss: %{y:.2f}%<extra></extra>",
    )
    fig.update_layout(
        **_title_layout("Share of annual forest loss (100% per year)"),
        legend=dict(title="Country", orientation="v", y=1, x=1.02, font=dict(size=10)),
        yaxis=dict(range=[0, 100], title="Share (%)"),
    )
    fig.update_xaxes(title_text="Year")
    _apply_chart_axes(fig, show_x_grid=False, show_y_grid=False)
    return fig


def _yoy_long(df: pd.DataFrame, max_countries: int = 10) -> pd.DataFrame | None:
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
        return None
    long = pd.concat(parts, ignore_index=True)
    top = long.groupby("country")["yoy_pct"].apply(lambda x: np.nanmean(np.abs(x))).sort_values(ascending=False)
    keep = top.head(max_countries).index
    return long[long["country"].isin(keep)]


def figure_yoy_growth(df: pd.DataFrame, max_countries: int = 10) -> go.Figure:
    long = _yoy_long(df, max_countries=max_countries)
    if long is None or long.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough years per country for year-over-year growth.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#8fa3b0"),
        )
        fig.update_layout(**_title_layout("Year-over-year change (%)"), xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    fig = px.line(
        long,
        x="year",
        y="yoy_pct",
        color="country",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>Year: %{x}<br>YoY change: %{y:.2f}%<extra></extra>",
        line=dict(width=1.8),
        marker=dict(size=5),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(180,190,185,0.5)", line_width=1)
    fig.update_layout(
        **_title_layout("Year-over-year change in forest loss (%)"),
        legend=dict(title="Country", y=1, x=1.02, font=dict(size=10)),
    )
    fig.update_yaxes(title_text="YoY change (%)")
    fig.update_xaxes(title_text="Year")
    _apply_chart_axes(fig, show_x_grid=False, show_y_grid=True)
    return fig


def _cumulative_long(df: pd.DataFrame, max_countries: int = 10) -> pd.DataFrame | None:
    rows = []
    for country, g in df.groupby("country"):
        s = g.groupby("year", as_index=False)["forest_loss_area"].sum().sort_values("year")
        s["cumulative"] = s["forest_loss_area"].cumsum()
        s["country"] = country
        rows.append(s[["year", "country", "cumulative"]])
    if not rows:
        return None
    long = pd.concat(rows, ignore_index=True)
    totals = long.groupby("country")["cumulative"].max().sort_values(ascending=False)
    keep = totals.head(max_countries).index
    return long[long["country"].isin(keep)]


def figure_cumulative(df: pd.DataFrame, max_countries: int = 10) -> go.Figure:
    long = _cumulative_long(df, max_countries=max_countries)
    if long is None or long.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data for cumulative plot.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#8fa3b0"),
        )
        fig.update_layout(**_title_layout("Cumulative forest loss"), xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    fig = px.line(
        long,
        x="year",
        y="cumulative",
        color="country",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Dark2,
    )
    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>Year: %{x}<br>Cumulative loss: %{y:,.4f}<extra></extra>",
        line=dict(width=2),
        marker=dict(size=5),
    )
    fig.update_layout(
        **_title_layout("Cumulative forest loss over time"),
        legend=dict(title="Country", y=1, x=1.02, font=dict(size=10)),
    )
    fig.update_yaxes(title_text="Cumulative forest loss area")
    fig.update_xaxes(title_text="Year")
    _apply_chart_axes(fig, show_x_grid=False, show_y_grid=True)
    return fig


def figure_country_year_heatmap(df: pd.DataFrame) -> go.Figure:
    mat = df.pivot_table(
        index="country",
        columns="year",
        values="forest_loss_area",
        aggfunc="sum",
    ).sort_index()
    fig = px.imshow(
        mat.values,
        x=mat.columns.astype(str),
        y=mat.index.astype(str),
        color_continuous_scale="Greens",
        labels=dict(color="Forest loss"),
        aspect="auto",
    )
    fig.update_traces(
        hovertemplate="Country: %{y}<br>Year: %{x}<br>Forest loss: %{z:,.4f}<extra></extra>",
    )
    fig.update_layout(**_title_layout("Forest loss heatmap (country × year)"))
    fig.update_xaxes(title_text="Year", side="bottom", tickangle=-45)
    fig.update_yaxes(title_text="Country")
    _apply_chart_axes(fig, show_x_grid=False, show_y_grid=False)
    return fig


def figure_country_trend(df: pd.DataFrame, country: str) -> go.Figure:
    cdf = df[df["country"] == country].groupby("year", as_index=False)["forest_loss_area"].sum()
    fig = px.line(
        cdf,
        x="year",
        y="forest_loss_area",
        markers=True,
        color_discrete_sequence=["#43a047"],
    )
    fig.update_traces(
        hovertemplate=f"<b>{country}</b><br>Year: %{{x}}<br>Forest loss: %{{y:,.4f}}<extra></extra>",
        line=dict(width=2.5),
    )
    fig.update_layout(**_title_layout(f"{country} — forest loss over time"), showlegend=False)
    fig.update_yaxes(title_text="Forest loss area")
    fig.update_xaxes(title_text="Year")
    _apply_chart_axes(fig, show_x_grid=False, show_y_grid=True)
    return fig


def figure_predictions(pred_df: pd.DataFrame, country: str) -> go.Figure:
    cdf = pred_df[pred_df["country"] == country]
    fig = px.line(
        cdf,
        x="year",
        y="predicted_forest_loss_area",
        markers=True,
        color_discrete_sequence=["#ffa726"],
    )
    fig.update_traces(
        hovertemplate=f"<b>{country} (predicted)</b><br>Year: %{{x}}<br>Predicted loss: %{{y:,.4f}}<extra></extra>",
        line=dict(width=2.5),
    )
    fig.update_layout(**_title_layout(f"{country} — predicted forest loss"), showlegend=False)
    fig.update_yaxes(title_text="Predicted forest loss area")
    fig.update_xaxes(title_text="Year")
    _apply_chart_axes(fig, show_x_grid=False, show_y_grid=True)
    return fig


def figure_model_evaluation(eval_df: pd.DataFrame | None, country: str) -> go.Figure:
    if eval_df is None or eval_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data to evaluate the model for this country.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#8fa3b0"),
        )
        fig.update_layout(**_title_layout(f"{country} — actual vs predicted"), xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    x = eval_df["actual"].to_numpy()
    y = eval_df["predicted"].to_numpy()
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    if lo == hi:
        hi = lo + 1e-9

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Test points",
            marker=dict(size=10, color="#66bb6a", line=dict(color="#1b5e20", width=1)),
            hovertemplate=f"<b>{country}</b><br>Actual: %{{x:,.4f}}<br>Predicted: %{{y:,.4f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="Perfect fit (y = x)",
            line=dict(color="#ffa726", dash="dash", width=2),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        **_title_layout(f"{country} — actual vs predicted (test set)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True,
    )
    fig.update_xaxes(title_text="Actual forest loss area")
    fig.update_yaxes(title_text="Predicted forest loss area", scaleanchor="x", scaleratio=1)
    _apply_chart_axes(fig, show_x_grid=False, show_y_grid=False)
    return fig
