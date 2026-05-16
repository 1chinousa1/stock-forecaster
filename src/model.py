from data_loader import load_ticker
from features import build_features
from evaluate import walk_forward_evaluate, backtest_strategy
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import shap
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 50)
    print("STEP 1: Loading data and building features")
    print("=" * 50)
    df = load_ticker("BTC-USD")
    featured = build_features(df)

    EXCLUDE = ["target", "Open", "High", "Low", "Close", "Volume"]
    feature_cols = [c for c in featured.columns if c not in EXCLUDE]
    X = featured[feature_cols]
    y = featured["target"]
    print(f"Samples: {len(X)}, Features: {len(feature_cols)}\n")

    print("=" * 50)
    print("STEP 2: Tuning hyperparameters with Optuna")
    print("=" * 50)
    print("This may take a few minutes...\n")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
        }
        model = XGBClassifier(**params)
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_idx, test_idx in tscv.split(X):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            proba = model.predict_proba(X.iloc[test_idx])[:, 1]
            scores.append(roc_auc_score(y.iloc[test_idx], proba))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print(f"Best AUC-ROC found: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}\n")

    print("=" * 50)
    print("STEP 3: Training final model with best parameters")
    print("=" * 50)
    best_model = XGBClassifier(
        **study.best_params,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )

    results = walk_forward_evaluate(best_model, X, y)

    print("\n" + "=" * 50)
    print("STEP 4: Calibrating confidence scores")
    print("=" * 50)
    train_cutoff = int(len(X) * 0.8)
    best_model.fit(X.iloc[:train_cutoff], y.iloc[:train_cutoff])
    calibrated = CalibratedClassifierCV(best_model, method="sigmoid", cv=5)
    calibrated.fit(X.iloc[train_cutoff:], y.iloc[train_cutoff:])

    calib_proba = calibrated.predict_proba(X.iloc[train_cutoff:])[:, 1]
    fraction_pos, mean_pred = calibration_curve(
        y.iloc[train_cutoff:], calib_proba, n_bins=10
    )
    plt.figure(figsize=(8, 6))
    plt.plot(mean_pred, fraction_pos, "s-", label="Calibrated Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    calib_path = os.path.join(PROJECT_ROOT, "models", "calibration_curve.png")
    plt.savefig(calib_path)
    print(f"Calibration curve saved to {calib_path}")

    print("\n" + "=" * 50)
    print("STEP 5: SHAP feature importance analysis")
    print("=" * 50)
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X.iloc[train_cutoff:])

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X.iloc[train_cutoff:], max_display=20, show=False)
    plt.tight_layout()
    shap_path = os.path.join(PROJECT_ROOT, "models", "shap_summary.png")
    plt.savefig(shap_path, dpi=150, bbox_inches="tight")
    print(f"SHAP summary saved to {shap_path}")

    print("\n" + "=" * 50)
    print("STEP 6: Backtesting strategy")
    print("=" * 50)
    bt = backtest_strategy(featured, results["all_predictions"])
    final_market = bt["cumulative_market"].iloc[-1]
    final_strategy = bt["cumulative_strategy"].iloc[-1]
    print(f"Buy & Hold return:     {final_market:.2f}x")
    print(f"Model strategy return: {final_strategy:.2f}x")

    plt.figure(figsize=(12, 5))
    bt[["cumulative_market", "cumulative_strategy"]].plot(figsize=(12, 5))
    plt.title("Backtest: Model Strategy vs Buy & Hold")
    plt.ylabel("Cumulative Return")
    plt.tight_layout()
    bt_path = os.path.join(PROJECT_ROOT, "models", "backtest_chart.png")
    plt.savefig(bt_path)
    print(f"Backtest chart saved to {bt_path}")

    print("\n" + "=" * 50)
    print("STEP 7: Saving model artifacts")
    print("=" * 50)
    models_dir = os.path.join(PROJECT_ROOT, "models")
    joblib.dump(calibrated, os.path.join(models_dir, "xgb_calibrated.joblib"))
    joblib.dump(best_model, os.path.join(models_dir, "xgb_base.joblib"))
    with open(os.path.join(models_dir, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)
    print(f"Saved calibrated model to {models_dir}/xgb_calibrated.joblib")
    print(f"Saved base model to {models_dir}/xgb_base.joblib")
    print(f"Saved feature columns to {models_dir}/feature_cols.json")

    print("\n" + "=" * 50)
    print("ALL DONE! Weeks 3-5 complete.")
    print("=" * 50)


if __name__ == "__main__":
    main()