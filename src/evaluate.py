import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report


def walk_forward_evaluate(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    auc_scores = []
    all_preds = []
    all_labels = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, proba)
        auc_scores.append(auc)
        all_preds.extend(proba)
        all_labels.extend(y_test)

        print(f"  Fold {fold + 1}: AUC-ROC = {auc:.4f}")

    mean_auc = np.mean(auc_scores)
    print(f"\nMean AUC-ROC across {n_splits} folds: {mean_auc:.4f}")

    return {
        "mean_auc": mean_auc,
        "fold_aucs": auc_scores,
        "all_predictions": np.array(all_preds),
        "all_labels": np.array(all_labels),
    }


def backtest_strategy(df, predictions, threshold=0.55):
    bt = df[["Close"]].copy().iloc[-len(predictions):]
    bt["prediction"] = predictions
    bt["signal"] = (bt["prediction"] >= threshold).astype(int)
    bt["market_return"] = bt["Close"].pct_change()
    bt["strategy_return"] = bt["signal"].shift(1) * bt["market_return"]
    bt["cumulative_market"] = (1 + bt["market_return"]).cumprod()
    bt["cumulative_strategy"] = (1 + bt["strategy_return"]).cumprod()
    return bt