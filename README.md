# Price Movement Forecaster

An ML-powered system that predicts whether a stock or cryptocurrency price will increase or decrease over a configurable time horizon, with a calibrated confidence score.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![ML](https://img.shields.io/badge/ML-XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Active-green)

## Live Demo

**[Launch the App](https://your-app-url.streamlit.app)**

## Features

- Real-time price data from Yahoo Finance for any stock or crypto ticker
- XGBoost classifier tuned with Optuna Bayesian hyperparameter optimization
- Walk-forward (time-series) cross-validation to prevent data leakage
- Calibrated probability outputs using Platt Scaling for meaningful confidence scores
- SHAP explainability showing which technical indicators drive each prediction
- Backtesting framework comparing model strategy vs buy-and-hold
- Interactive Streamlit dashboard with candlestick charts and configurable thresholds

## How It Works

1. **Data Pipeline** — Downloads OHLCV price data via the `yfinance` API
2. **Feature Engineering** — Computes 82 technical indicators (RSI, MACD, Bollinger Bands, etc.) using the `ta` library
3. **Model Training** — Trains an XGBoost classifier with Optuna-tuned hyperparameters
4. **Calibration** — Applies Platt Scaling so the output probability reflects true statistical confidence
5. **Prediction** — Outputs a directional forecast with a percentage confidence score
6. **Explanation** — Uses SHAP values to show which features contributed to the prediction

## Tech Stack

- **Data:** yfinance, pandas, NumPy
- **Features:** ta (Technical Analysis library)
- **ML:** XGBoost, scikit-learn, Optuna
- **Explainability:** SHAP
- **App:** Streamlit, Plotly
- **Language:** Python 3.10+

## Project Structure