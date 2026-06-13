"""
app_sales_forecast.py
======================
Sales KPI Dashboard & 30-Day Forecast — a self-contained Streamlit application.

What this app does
------------------
1.  Generates 2+ years of realistic daily sales data (trend + weekly & yearly
    seasonality + promotions + noise) across several product categories.
2.  Cleans the data with several clearly commented steps.
3.  Surfaces headline KPIs with ``st.metric`` (total sales, profit, averages).
4.  Plots interactive Plotly time-series of historical performance.
5.  Produces a 30-day forecast: it tries Facebook **Prophet** first and
    gracefully falls back to a **scipy**-based exponential-smoothing model if
    Prophet is not installed.

Run it with:
    streamlit run app_sales_forecast.py

Only the following libraries are used: streamlit, pandas, numpy, plotly,
scipy and (optionally) prophet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize_scalar

# Prophet is optional. We detect availability once at import time so the rest of
# the app can branch cleanly without repeated try/except blocks.
try:
    from prophet import Prophet  # type: ignore

    PROPHET_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    PROPHET_AVAILABLE = False

# --------------------------------------------------------------------------- #
# Page configuration                                                          #
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Sales KPI Dashboard & Forecast",
    page_icon="📈",
    layout="wide",
)

# Product categories and their relative baseline sales weight / margin.
CATEGORIES = {
    "Electronics": {"base": 1800, "margin": 0.18},
    "Clothing": {"base": 1200, "margin": 0.45},
    "Home & Garden": {"base": 900, "margin": 0.35},
    "Groceries": {"base": 2200, "margin": 0.12},
    "Toys": {"base": 600, "margin": 0.40},
}


# --------------------------------------------------------------------------- #
# 1. Synthetic data generation                                                #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Generating synthetic sales history…")
def generate_sales_data(n_days: int = 760, seed: int = 7) -> pd.DataFrame:
    """Create a daily sales table spanning ``n_days`` ending today.

    Each category gets its own signal built from:
      * a gentle upward growth trend,
      * weekly seasonality (weekends sell more),
      * yearly seasonality (holiday-season uplift),
      * random promotional spikes, and
      * multiplicative noise.

    Returns
    -------
    pandas.DataFrame
        Long-format table: one row per (date, category).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")
    day_index = np.arange(n_days)

    records = []
    for category, cfg in CATEGORIES.items():
        base = cfg["base"]

        # Linear growth trend (~30% over the full window).
        trend = base * (1 + 0.0004 * day_index)

        # Weekly seasonality: Saturday/Sunday get a boost.
        weekday = dates.dayofweek.to_numpy()
        weekly = np.where(weekday >= 5, 1.25, 1.0)

        # Yearly seasonality: peak around day-of-year ~330 (Nov/Dec holidays).
        doy = dates.dayofyear.to_numpy()
        yearly = 1.0 + 0.20 * np.cos(2 * np.pi * (doy - 330) / 365)

        # Random promotional spikes on ~4% of days.
        promo = np.where(rng.random(n_days) < 0.04, rng.uniform(1.3, 1.8, n_days), 1.0)

        # Multiplicative noise.
        noise = rng.normal(1.0, 0.08, n_days)

        units = trend * weekly * yearly * promo * noise
        units = np.clip(units, 0, None)  # never negative

        avg_price = rng.uniform(15, 60)
        revenue = np.round(units * avg_price / 100, 2)
        profit = np.round(revenue * cfg["margin"], 2)

        records.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "category": category,
                    "units_sold": np.round(units).astype(int),
                    "revenue": revenue,
                    "profit": profit,
                }
            )
        )

    data = pd.concat(records, ignore_index=True)

    # --- Inject some "dirtiness" so cleaning is meaningful -------------------
    # (a) A few duplicated rows.
    data = pd.concat([data, data.sample(40, random_state=seed)], ignore_index=True)
    # (b) Some missing revenue values.
    miss_idx = rng.choice(len(data), size=int(0.02 * len(data)), replace=False)
    data.loc[miss_idx, "revenue"] = np.nan
    # (c) A couple of impossible negative revenue entries.
    neg_idx = rng.choice(len(data), size=10, replace=False)
    data.loc[neg_idx, "revenue"] = -data.loc[neg_idx, "revenue"].abs()

    return data


# --------------------------------------------------------------------------- #
# 2. Data cleaning                                                            #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Cleaning sales data…")
def clean_sales_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw sales table with transparent, commented steps.

    Cleaning steps:
      1. Ensure the ``date`` column is a proper datetime.
      2. Drop duplicate (date, category) rows.
      3. Remove impossible negative revenue rows.
      4. Impute missing revenue from units sold and re-derive profit.
    """
    df = raw.copy()

    # --- Cleaning step 1: enforce datetime dtype -----------------------------
    df["date"] = pd.to_datetime(df["date"])

    # --- Cleaning step 2: remove duplicate day/category records --------------
    df = df.drop_duplicates(subset=["date", "category"]).reset_index(drop=True)

    # --- Cleaning step 3: drop physically impossible negatives ---------------
    df = df[df["revenue"].fillna(0) >= 0].reset_index(drop=True)

    # --- Cleaning step 4: impute missing revenue, then re-derive profit ------
    # Use each category's median revenue-per-unit to backfill missing revenue.
    df["rev_per_unit"] = df["revenue"] / df["units_sold"].replace(0, np.nan)
    cat_rev_per_unit = df.groupby("category")["rev_per_unit"].transform("median")
    df["revenue"] = df["revenue"].fillna(df["units_sold"] * cat_rev_per_unit)
    df = df.drop(columns="rev_per_unit")

    # Re-derive profit where it is missing using the category margin map.
    margins = {c: cfg["margin"] for c, cfg in CATEGORIES.items()}
    df["profit"] = df["profit"].fillna(
        df["revenue"] * df["category"].map(margins)
    )

    return df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. Forecasting                                                              #
# --------------------------------------------------------------------------- #
def _exponential_smoothing_forecast(
    series: pd.Series, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """scipy-based Holt linear (double exponential) smoothing fallback.

    The smoothing factors ``alpha`` (level) and ``beta`` (trend) are tuned by
    minimising the one-step-ahead squared error using ``scipy.optimize``.

    Returns
    -------
    (forecast, fitted)
        ``forecast`` holds ``horizon`` future points; ``fitted`` holds the
        in-sample one-step-ahead predictions (handy for plotting the fit).
    """
    values = series.to_numpy(dtype=float)
    n = len(values)

    def sse_for_alpha(alpha: float, beta: float) -> tuple[float, np.ndarray]:
        """Return (sum of squared errors, fitted values) for given factors."""
        level = values[0]
        trend = values[1] - values[0]
        fitted = np.zeros(n)
        fitted[0] = level
        sse = 0.0
        for t in range(1, n):
            forecast = level + trend
            fitted[t] = forecast
            error = values[t] - forecast
            sse += error ** 2
            level = alpha * values[t] + (1 - alpha) * (forecast)
            trend = beta * (level - (forecast - trend)) + (1 - beta) * trend
        return sse, fitted

    # Optimise alpha for a fixed, mild trend factor (keeps it fast & stable).
    beta = 0.1
    result = minimize_scalar(
        lambda a: sse_for_alpha(a, beta)[0], bounds=(0.01, 0.99), method="bounded"
    )
    best_alpha = float(result.x)
    _, fitted = sse_for_alpha(best_alpha, beta)

    # Re-run to obtain the final level & trend, then extrapolate.
    level = values[0]
    trend = values[1] - values[0]
    for t in range(1, n):
        forecast = level + trend
        level = best_alpha * values[t] + (1 - best_alpha) * forecast
        trend = beta * (level - (level - trend)) + (1 - beta) * trend

    forecast = np.array([level + (h + 1) * trend for h in range(horizon)])
    forecast = np.clip(forecast, 0, None)
    return forecast, fitted


@st.cache_data(show_spinner="Building 30-day forecast…")
def make_forecast(daily: pd.DataFrame, horizon: int = 30) -> dict:
    """Forecast total daily revenue ``horizon`` days ahead.

    Tries Prophet first (richer seasonality handling) and falls back to the
    scipy exponential-smoothing routine if Prophet is unavailable or errors.

    ``daily`` must have columns ``date`` and ``revenue`` already aggregated to
    one row per day.
    """
    history = daily[["date", "revenue"]].rename(columns={"date": "ds", "revenue": "y"})
    future_dates = pd.date_range(
        start=history["ds"].max() + pd.Timedelta(days=1), periods=horizon, freq="D"
    )

    if PROPHET_AVAILABLE:
        try:
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
            )
            model.fit(history)
            future = model.make_future_dataframe(periods=horizon)
            forecast = model.predict(future)
            future_part = forecast.tail(horizon)
            return {
                "method": "Prophet",
                "future_dates": future_part["ds"].to_numpy(),
                "forecast": future_part["yhat"].to_numpy(),
                "lower": future_part["yhat_lower"].to_numpy(),
                "upper": future_part["yhat_upper"].to_numpy(),
            }
        except Exception:
            # Any Prophet runtime failure quietly drops to the fallback below.
            pass

    # ---- scipy fallback -----------------------------------------------------
    forecast, _ = _exponential_smoothing_forecast(history["y"], horizon)
    # Build a simple +/- confidence band from the historical residual std.
    resid_std = float(history["y"].diff().std())
    return {
        "method": "Exponential Smoothing (scipy)",
        "future_dates": future_dates.to_numpy(),
        "forecast": forecast,
        "lower": np.clip(forecast - 1.96 * resid_std, 0, None),
        "upper": forecast + 1.96 * resid_std,
    }


# --------------------------------------------------------------------------- #
# 4. Plotly helpers                                                           #
# --------------------------------------------------------------------------- #
def plot_history(daily: pd.DataFrame) -> go.Figure:
    """Line chart of total daily revenue with a 30-day moving average."""
    daily = daily.sort_values("date")
    moving_avg = daily["revenue"].rolling(30, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["date"], y=daily["revenue"], name="Daily revenue",
            mode="lines", line=dict(color="#93c5fd", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date"], y=moving_avg, name="30-day moving avg",
            mode="lines", line=dict(color="#1d4ed8", width=3),
        )
    )
    fig.update_layout(
        title="Historical Daily Revenue",
        xaxis_title="Date",
        yaxis_title="Revenue ($)",
        hovermode="x unified",
    )
    return fig


def plot_forecast(daily: pd.DataFrame, fc: dict) -> go.Figure:
    """Overlay the forecast (and confidence band) on recent history."""
    recent = daily.sort_values("date").tail(90)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent["date"], y=recent["revenue"], name="History",
            mode="lines", line=dict(color="#1d4ed8"),
        )
    )
    # Confidence band (drawn as an upper line + filled lower line).
    fig.add_trace(
        go.Scatter(
            x=fc["future_dates"], y=fc["upper"], name="Upper bound",
            mode="lines", line=dict(width=0), showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc["future_dates"], y=fc["lower"], name="Confidence band",
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(220,38,38,0.15)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc["future_dates"], y=fc["forecast"], name="Forecast",
            mode="lines", line=dict(color="#dc2626", width=3, dash="dash"),
        )
    )
    fig.update_layout(
        title=f"30-Day Revenue Forecast · {fc['method']}",
        xaxis_title="Date",
        yaxis_title="Revenue ($)",
        hovermode="x unified",
    )
    return fig


def plot_category_breakdown(filtered: pd.DataFrame) -> go.Figure:
    """Stacked bar chart of monthly revenue split by category."""
    monthly = (
        filtered.assign(month=filtered["date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["month", "category"], as_index=False)["revenue"]
        .sum()
    )
    fig = px.bar(
        monthly, x="month", y="revenue", color="category",
        labels={"month": "Month", "revenue": "Revenue ($)"},
    )
    fig.update_layout(title="Monthly Revenue by Category", barmode="stack")
    return fig


def plot_weekday_profile(filtered: pd.DataFrame) -> go.Figure:
    """Average revenue by day of week — reveals the weekly seasonality."""
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_weekday = (
        filtered.assign(weekday=filtered["date"].dt.day_name().str[:3])
        .groupby("weekday", as_index=False)["revenue"]
        .mean()
    )
    by_weekday["weekday"] = pd.Categorical(by_weekday["weekday"], order, ordered=True)
    by_weekday = by_weekday.sort_values("weekday")
    fig = px.bar(
        by_weekday, x="weekday", y="revenue", color="revenue",
        color_continuous_scale="Tealgrn",
        labels={"weekday": "Day of week", "revenue": "Avg revenue ($)"},
    )
    fig.update_layout(title="Average Revenue by Day of Week", coloraxis_showscale=False)
    return fig


# --------------------------------------------------------------------------- #
# 5. Sidebar controls                                                         #
# --------------------------------------------------------------------------- #
def build_sidebar(clean: pd.DataFrame) -> dict:
    """Render the sidebar filters and return the chosen values."""
    st.sidebar.header("🔧 Filters")

    min_date = clean["date"].min().date()
    max_date = clean["date"].max().date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    selected_categories = st.sidebar.multiselect(
        "Product categories",
        options=list(CATEGORIES.keys()),
        default=list(CATEGORIES.keys()),
    )

    st.sidebar.markdown("---")
    horizon = st.sidebar.slider("Forecast horizon (days)", 7, 60, 30, step=1)

    st.sidebar.caption(
        f"Forecast engine: **{'Prophet' if PROPHET_AVAILABLE else 'scipy fallback'}**"
    )
    return {
        "date_range": date_range,
        "categories": selected_categories,
        "horizon": horizon,
    }


def apply_filters(clean: pd.DataFrame, controls: dict) -> pd.DataFrame:
    """Filter the cleaned data by the sidebar date range & categories."""
    df = clean.copy()
    date_range = controls["date_range"]

    # date_input returns a single date until both ends are picked.
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        df = df[(df["date"] >= start) & (df["date"] <= end)]

    if controls["categories"]:
        df = df[df["category"].isin(controls["categories"])]

    return df


# --------------------------------------------------------------------------- #
# 6. Main application body                                                     #
# --------------------------------------------------------------------------- #
def main() -> None:
    """Assemble the full dashboard."""
    st.title("📈 Sales KPI Dashboard & 30-Day Forecast")
    st.markdown(
        "Explore two years of synthetic multi-category sales data and project "
        "the next month of revenue."
    )

    raw = generate_sales_data()
    clean = clean_sales_data(raw)
    controls = build_sidebar(clean)
    filtered = apply_filters(clean, controls)

    if filtered.empty:
        st.warning("No data matches the current filters. Widen your selection.")
        st.stop()

    # --- KPIs ----------------------------------------------------------------
    daily = filtered.groupby("date", as_index=False).agg(
        revenue=("revenue", "sum"),
        profit=("profit", "sum"),
        units=("units_sold", "sum"),
    )
    total_sales = daily["revenue"].sum()
    total_profit = daily["profit"].sum()
    avg_daily = daily["revenue"].mean()
    total_units = int(daily["units"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total revenue", f"${total_sales:,.0f}")
    c2.metric("Total profit", f"${total_profit:,.0f}")
    c3.metric("Avg daily revenue", f"${avg_daily:,.0f}")
    c4.metric("Units sold", f"{total_units:,}")

    st.markdown("---")

    # --- Historical time series ----------------------------------------------
    st.subheader("🕒 Historical Performance")
    st.plotly_chart(plot_history(daily), use_container_width=True)

    cat_col, week_col = st.columns(2)
    with cat_col:
        st.plotly_chart(plot_category_breakdown(filtered), use_container_width=True)
    with week_col:
        st.plotly_chart(plot_weekday_profile(filtered), use_container_width=True)

    st.markdown("---")

    # --- Forecast ------------------------------------------------------------
    st.subheader("🔮 Revenue Forecast")
    fc = make_forecast(daily, controls["horizon"])
    st.plotly_chart(plot_forecast(daily, fc), use_container_width=True)

    forecast_total = float(np.sum(fc["forecast"]))
    f1, f2, f3 = st.columns(3)
    f1.metric(f"Forecast revenue ({controls['horizon']}d)", f"${forecast_total:,.0f}")
    f2.metric("Avg forecast / day", f"${forecast_total / controls['horizon']:,.0f}")
    f3.metric("Forecast engine", fc["method"])

    with st.expander("🔎 Preview the cleaned dataset"):
        st.dataframe(filtered.head(100), use_container_width=True)
        st.caption(f"Showing 100 of {len(filtered):,} filtered rows.")


if __name__ == "__main__":
    main()
