import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import joblib
import json
import shap
import sys
import os
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from features import build_features

st.set_page_config(page_title="Price Movement Forecaster", layout="wide")
st.title("Price Movement Forecaster")
st.caption("ML-powered directional prediction with calibrated confidence scores.")


@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(PROJECT_ROOT, "models", "xgb_calibrated.joblib"))
    base = joblib.load(os.path.join(PROJECT_ROOT, "models", "xgb_base.joblib"))
    with open(os.path.join(PROJECT_ROOT, "models", "feature_cols.json")) as f:
        feature_cols = json.load(f)
    return model, base, feature_cols


model, base_model, feature_cols = load_model()

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker Symbol", value="BTC-USD").upper()
    forward_days = st.slider("Forecast Horizon (days)", 1, 20, 5)
    threshold = st.slider("Confidence Threshold (%)", 50, 90, 55) / 100
    run = st.button("Run Forecast", type="primary")

if run:
    with st.spinner(f"Fetching data for {ticker}..."):
        raw = yf.download(ticker, period="2y", auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

    if raw.empty:
        st.error(f"No data found for '{ticker}'. Check the symbol and try again.")
    else:
        df = build_features(raw, forward_days=forward_days)
        available = [c for c in feature_cols if c in df.columns]
        X_latest = df[available].iloc[[-1]]

        proba = model.predict_proba(X_latest)[0, 1]
        direction = "Likely to Increase" if proba >= 0.5 else "Likely to Decrease"
        confidence = proba if proba >= 0.5 else 1 - proba

        col1, col2, col3 = st.columns(3)
        col1.metric("Ticker", ticker)
        col2.metric("Prediction", direction)
        col3.metric("Confidence", f"{confidence * 100:.1f}%")

        if confidence < threshold:
            st.warning(
                f"Confidence of {confidence * 100:.1f}% is below your "
                f"threshold of {threshold * 100:.0f}%. No actionable signal."
            )
        else:
            st.success(
                f"Signal active: {confidence * 100:.1f}% confidence "
                f"that price will {'increase' if proba >= 0.5 else 'decrease'} "
                f"over the next {forward_days} days."
            )

        st.subheader("Price History")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=raw.index, open=raw["Open"], high=raw["High"],
            low=raw["Low"], close=raw["Close"], name=ticker
        ))
        fig.update_layout(xaxis_rangeslider_visible=False, height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("What Is Driving This Prediction?")
        try:
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(X_latest)
            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(1, -1)

            feature_importance = pd.DataFrame({
                "feature": X_latest.columns,
                "shap_value": shap_values[0]
            })
            feature_importance["abs_shap"] = feature_importance["shap_value"].abs()
            feature_importance = feature_importance.nlargest(15, "abs_shap")
            feature_importance = feature_importance.sort_values("shap_value")

            colors = ["#ef4444" if v < 0 else "#22c55e"
                      for v in feature_importance["shap_value"]]

            fig_shap = go.Figure(go.Bar(
                x=feature_importance["shap_value"],
                y=feature_importance["feature"],
                orientation="h",
                marker_color=colors,
            ))
            fig_shap.update_layout(
                height=450,
                xaxis_title="Impact on Prediction",
                yaxis_title="",
                template="plotly_dark",
                annotations=[
                    dict(x=0.02, y=1.06, xref="paper", yref="paper",
                         text="<b>Green</b> = pushes toward Increase",
                         showarrow=False, font=dict(color="#22c55e", size=12)),
                    dict(x=0.98, y=1.06, xref="paper", yref="paper",
                         text="<b>Red</b> = pushes toward Decrease",
                         showarrow=False, font=dict(color="#ef4444", size=12),
                         xanchor="right"),
                ]
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as e:
            st.info(f"SHAP visualization unavailable: {e}")

        st.divider()
        st.caption(
            "This tool is for educational purposes only and does not "
            "constitute financial advice. Past patterns do not guarantee "
            "future results."
        )