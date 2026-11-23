# Streamlit dashboard for the Gampaha ensemble-learning project.
# Generated with Claude (Anthropic).
# Run the notebook first: it creates the ./artifacts folder this app reads.
from datetime import timedelta
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Gampaha Weather: Ensemble Learning", page_icon="🌦️", layout="wide")

ART = Path(__file__).parent / "artifacts"
FAMILY_COLORS = {"Baseline": "#9aa0a6", "Bagging": "#2e7d32", "Boosting": "#e65100",
                 "Neural net": "#6a1b9a", "Hybrid": "#0277bd", "Stacking": "#c62828"}

REQUIRED = ["models.joblib", "metrics.csv", "features.csv", "test_predictions.csv", "importances.csv"]
missing = [f for f in REQUIRED if not (ART / f).exists()]
if missing:
    st.error("Missing files in the `artifacts/` folder: " + ", ".join(missing) +
             ". Run the notebook first and copy its `artifacts` folder next to `app.py`.")
    st.stop()


@st.cache_resource
def load_bundle():
    return joblib.load(ART / "models.joblib")


@st.cache_data
def load_tables():
    metrics = pd.read_csv(ART / "metrics.csv")
    feats = pd.read_csv(ART / "features.csv", parse_dates=["date"]).set_index("date")
    preds = pd.read_csv(ART / "test_predictions.csv", parse_dates=["date"]).set_index("date")
    imps = pd.read_csv(ART / "importances.csv")
    return metrics, feats, preds, imps


bundle = load_bundle()
metrics, feats, preds, imps = load_tables()
meta = bundle["meta"]
family_of = dict(zip(metrics["model"], metrics["family"]))
best = metrics.sort_values("RMSE").iloc[0]
persist_rmse = metrics.loc[metrics["model"] == meta["PERSIST"], "RMSE"].iloc[0]


def predict_all(X, arima_pred):
    """Forecast for every model in the comparison, from a one-row (or more) feature table."""
    out = {name: m.predict(X) for name, m in bundle["models"].items()}
    out[meta["PERSIST"]] = X["tmean_lag0"].values
    out[meta["ARIMA_ALONE"]] = arima_pred
    out[meta["HYB_ANN"]] = (out["Random Forest"] + out["ANN (MLP)"]) / 2
    out[meta["HYB_ARIMA"]] = arima_pred + bundle["hybrid_residual_model"].predict(X)
    return out


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Gampaha weather")
st.sidebar.caption("Next-day mean temperature, forecast with ensemble models")
page = st.sidebar.radio("Section", ["Overview", "Model comparison", "Predictions over time",
                                    "Feature importance", "Try it: next-day forecast"])
st.sidebar.divider()
st.sidebar.caption("Code generated with Claude (Anthropic).  \n"
                   "Weather data: [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0).")

# ------------------------------------------------------------------ pages
if page == "Overview":
    st.title("Can an ensemble beat \"tomorrow = today\"?")
    st.write("Forecasting **tomorrow's mean temperature in Gampaha, Sri Lanka** from 11 years of daily weather, "
             "and comparing bagging, boosting, stacking and hybrid models on a held-out test period.")
    st.info("**AI disclosure:** the code for this dashboard and the notebook was generated with Claude (Anthropic).")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", best["model"])
    c2.metric("Best RMSE (deg C)", f"{best['RMSE']:.3f}")
    c3.metric("vs. persistence baseline", f"{(1 - best['RMSE'] / persist_rmse) * 100:.1f}% lower RMSE")
    c4.metric("Test days", f"{len(preds):,}")

    st.subheader("The four ensemble ideas")
    a, b, c, d = st.columns(4)
    a.markdown("**Bagging**  \nMany models on random resamples, averaged. Cuts *variance*.  \n"
               "*Random Forest, Bagged Trees*")
    b.markdown("**Boosting**  \nModels added one by one, each fixing the last one's errors. Cuts *bias*.  \n"
               "*Gradient Boosting, XGBoost, LightGBM, CatBoost*")
    c.markdown("**Stacking**  \nDifferent model types, with a meta-model learning how to weigh them.  \n"
               "*RF + XGBoost + CatBoost + ANN, Ridge on top*")
    d.markdown("**Hybrid**  \nA statistical model plus machine learning.  \n"
               "*ARIMA + RF on its residuals; RF + ANN average*")

    st.subheader("Data and method")
    st.markdown(
        f"- **Location:** {meta['location']} ({meta['lat']:.2f} N, {meta['lon']:.2f} E)\n"
        f"- **Source:** Open-Meteo Historical Weather API (ERA5 reanalysis), daily values\n"
        f"- **Target:** next day's mean temperature (deg C)\n"
        f"- **Split by time:** trained up to {meta['train_end']}, tested from {meta['test_start']} "
        f"(a random split would leak the future)\n"
        f"- **Metrics:** MAE, RMSE, R2 on the test period")

elif page == "Model comparison":
    st.title("Model comparison")
    metric = st.radio("Metric", ["RMSE", "MAE", "R2"], horizontal=True)
    ascending = metric != "R2"
    d = metrics.sort_values(metric, ascending=ascending)
    fig = px.bar(d, x=metric, y="model", color="family", orientation="h",
                 color_discrete_map=FAMILY_COLORS, category_orders={"model": d["model"].tolist()})
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_layout(height=480, legend_title_text="Family", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption("RMSE and MAE are in deg C (lower is better); R2 is higher-is-better. "
               "Differences of a few hundredths of a degree between top models are small, "
               "so read the ranking together with the notebook's discussion.")
    st.dataframe(d.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}", "fit_seconds": "{:.1f}"}),
                 hide_index=True, width="stretch")

    st.subheader("Best of each family")
    fam_best = metrics.loc[metrics.groupby("family")["RMSE"].idxmin()].sort_values("RMSE")
    st.dataframe(fam_best[["family", "model", "RMSE", "MAE", "R2"]].style.format(
        {"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}"}), hide_index=True, width="stretch")

    weights = meta.get("meta_weights")
    if weights:
        st.subheader("What the stacking meta-model trusts")
        w = pd.DataFrame({"base model": list(weights), "Ridge weight": list(weights.values())})
        st.plotly_chart(px.bar(w, x="base model", y="Ridge weight").update_layout(height=300), width="stretch")
        st.caption("A large weight on one model means the meta-model mostly copies it; "
                   "small gains from stacking usually mean the base models make similar mistakes.")

elif page == "Predictions over time":
    st.title("Predictions over time (test period)")
    names = [c for c in preds.columns if c != "actual"]
    default = [n for n in [best["model"], meta["PERSIST"]] if n in names]
    selected = st.multiselect("Models to show", names, default=default)
    lo_d, hi_d = preds.index.min().to_pydatetime(), preds.index.max().to_pydatetime()
    start, end = st.slider("Date range", min_value=lo_d, max_value=hi_d,
                           value=(max(lo_d, hi_d - timedelta(days=120)), hi_d), format="DD MMM YYYY")
    view = preds.loc[start:end]
    fig = go.Figure()
    fig.add_scatter(x=view.index, y=view["actual"], name="Actual", line=dict(color="black", width=2.5))
    for n in selected:
        fig.add_scatter(x=view.index, y=view[n], name=n, mode="lines",
                        line=dict(color=FAMILY_COLORS.get(family_of.get(n), None), width=1.6))
    fig.update_layout(height=450, yaxis_title="Mean temperature (deg C)", xaxis_title="Day being forecast",
                      margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    if selected:
        focus = st.selectbox("Error distribution for", selected)
        err = (preds["actual"] - preds[focus]).rename("actual - predicted (deg C)")
        st.plotly_chart(px.histogram(err, nbins=40).update_layout(height=300, showlegend=False),
                        width="stretch")
        st.caption(f"Mean error {err.mean():+.3f} deg C, standard deviation {err.std():.3f} deg C "
                   "(whole test period).")

elif page == "Feature importance":
    st.title("Which inputs matter?")
    options = [c for c in imps.columns if c != "feature"]
    choice = st.selectbox("Importance measure", options)
    top_n = st.slider("Number of features", 5, 25, 15)
    top = imps[["feature", choice]].sort_values(choice, ascending=False).head(top_n).iloc[::-1]
    fig = px.bar(top, x=choice, y="feature", orientation="h")
    fig.update_layout(height=max(350, 24 * top_n), margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption("Names read as variable_lagN: e.g. tmean_lag1 = mean temperature one day before 'today'. "
               "doy_sin/doy_cos encode the time of year; *_roll7/30 are rolling averages (rain: totals). "
               "Importances are normalised to sum to 1 within each measure.")

else:
    st.title("Try it: next-day forecast")
    st.write("Pick a day in the test period. Every model forecasts the **next day's** mean temperature "
             "from that day's weather and recent history.")
    test_feats = feats[feats["split"] == "test"]
    day = st.date_input("Today's date", value=test_feats.index.max().date(),
                        min_value=test_feats.index.min().date(), max_value=test_feats.index.max().date())
    ts = pd.Timestamp(day)
    if ts not in test_feats.index:
        st.warning("No data for that date.")
        st.stop()
    row = test_feats.loc[[ts]]
    X = row[bundle["feature_cols"]].copy()

    with st.expander("What-if: change today's weather"):
        use_override = st.checkbox("Override today's values")
        t_lo, t_hi = float(feats["tmean_lag0"].min()), float(feats["tmean_lag0"].max())
        t0 = st.slider("Today's mean temperature (deg C)", t_lo, t_hi, float(X["tmean_lag0"].iloc[0]), 0.1)
        t1 = st.slider("Yesterday's mean temperature (deg C)", t_lo, t_hi, float(X["tmean_lag1"].iloc[0]), 0.1)
        r0 = st.slider("Today's rainfall (mm)", 0.0, float(feats["rain_lag0"].max()),
                       float(X["rain_lag0"].iloc[0]), 0.5)
        st.caption("Rolling averages and the ARIMA forecast are not recomputed, so keep changes modest.")
    if use_override:
        X["tmean_lag0"], X["tmean_lag1"], X["rain_lag0"] = t0, t1, r0

    out = predict_all(X, row["arima_pred"].values)
    table = pd.DataFrame({"model": list(out), "forecast": [float(v[0]) for v in out.values()]})
    table["family"] = table["model"].map(family_of)
    actual = float(row["target"].iloc[0])
    table["abs. error vs actual"] = (table["forecast"] - actual).abs()

    c1, c2 = st.columns(2)
    c1.metric("Today's mean temperature", f"{float(X['tmean_lag0'].iloc[0]):.1f} deg C")
    c2.metric("Actual next-day mean" + (" (original data)" if use_override else ""), f"{actual:.2f} deg C")

    fig = px.bar(table.sort_values("forecast"), x="forecast", y="model", color="family", orientation="h",
                 color_discrete_map=FAMILY_COLORS)
    lo = table["forecast"].min() - 0.5
    hi = table["forecast"].max() + 0.5
    fig.update_xaxes(range=[min(lo, actual - 0.3), max(hi, actual + 0.3)], title="Forecast mean temperature (deg C)")
    fig.add_vline(x=actual, line_dash="dash", line_color="black", annotation_text="actual")
    fig.update_yaxes(title=None)
    fig.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption("The x-axis is zoomed to the forecasts, so bar lengths exaggerate small differences; read the values below.")
    st.dataframe(table.sort_values("abs. error vs actual").style.format(
        {"forecast": "{:.2f}", "abs. error vs actual": "{:.2f}"}), hide_index=True, width="stretch")
