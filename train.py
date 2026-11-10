"""Train Random Forest and XGBoost match-outcome models (home win / draw / away win).

* Split is by time: tune and fit on older seasons, test on the most recent ones (never shuffled).
* Hyperparameters are picked with TimeSeriesSplit cross-validation on the training seasons (log loss).
* The model with the better CV log loss is saved to models/model.joblib for predict.py.

Usage:  python train.py [--test-from 2025]
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from features import LABELS, build_features, feature_columns, load_matches

ROOT = Path(__file__).parent
MIN_PRIOR = 3          # both teams need at least this many earlier matches in the data
WARMUP_SEASON = 2019   # first season only warms up Elo / rolling stats, not used for training


def brier(y, proba):
    onehot = np.eye(3)[y]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def evaluate(name, y, proba):
    pred = proba.argmax(1)
    return {
        "model": name,
        "accuracy": round(accuracy_score(y, pred), 4),
        "log_loss": round(log_loss(y, proba, labels=[0, 1, 2]), 4),
        "brier": round(brier(y, proba), 4),
        "macro_f1": round(f1_score(y, pred, average="macro"), 4),
        "pred_share": {LABELS[k]: round(float((pred == k).mean()), 3) for k in range(3)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-from", type=int, default=2025, help="first season held out for testing")
    args = ap.parse_args()

    df = build_features(load_matches(ROOT / "data" / "matches.csv"))
    cols = feature_columns()
    data = df[df.played & (df.season > WARMUP_SEASON)
              & (df.home_n_prior >= MIN_PRIOR) & (df.away_n_prior >= MIN_PRIOR)]
    train, test = data[data.season < args.test_from], data[data.season >= args.test_from]
    Xtr, ytr, Xte, yte = train[cols], train.target.values, test[cols], test.target.values
    print(f"train: {len(train)} matches ({train.date.min():%Y-%m} to {train.date.max():%Y-%m})")
    print(f"test : {len(test)} matches ({test.date.min():%Y-%m} to {test.date.max():%Y-%m})")
    print("train outcome share:", dict(zip(LABELS, np.bincount(ytr, minlength=3) / len(ytr))))

    cv = TimeSeriesSplit(n_splits=4)
    searches = {
        "random_forest": GridSearchCV(
            RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=42),
            {"max_depth": [6, 10, None], "min_samples_leaf": [10, 30, 60], "max_features": ["sqrt", 0.3]},
            cv=cv, scoring="neg_log_loss", n_jobs=1),
        "xgboost": GridSearchCV(
            XGBClassifier(objective="multi:softprob", learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.7, eval_metric="mlogloss", n_jobs=-1, random_state=42),
            {"max_depth": [2, 3, 4], "n_estimators": [200, 400, 700], "min_child_weight": [1, 10]},
            cv=cv, scoring="neg_log_loss", n_jobs=1),
    }

    results, fitted = [], {}
    # baselines: class frequencies, always-home-win, and an Elo + home-advantage logistic regression
    prior = np.bincount(ytr, minlength=3) / len(ytr)
    results.append(evaluate("baseline: class frequencies", yte, np.tile(prior, (len(yte), 1))))
    results[-1]["accuracy"] = round(float((yte == 2).mean()), 4)
    results[-1]["model"] = "baseline: always home win (probs = class freq.)"
    elo_cols = ["elo_diff", "home_adv"]
    elo_lr = LogisticRegression(max_iter=1000).fit(Xtr[elo_cols], ytr)
    results.append(evaluate("baseline: Elo-only logistic regression", yte, elo_lr.predict_proba(Xte[elo_cols])))

    for name, search in searches.items():
        print(f"\ntuning {name} ...")
        search.fit(Xtr, ytr)
        print(f"  best params {search.best_params_}  CV log loss {-search.best_score_:.4f}")
        fitted[name] = (search.best_estimator_, -search.best_score_, search.best_params_)
        r = evaluate(name, yte, search.best_estimator_.predict_proba(Xte))
        r["cv_log_loss"] = round(-search.best_score_, 4)
        r["best_params"] = search.best_params_
        results.append(r)

    table = pd.DataFrame(results).set_index("model")
    print("\n=== held-out test results (lower log loss / brier is better) ===")
    print(table[["accuracy", "log_loss", "brier", "macro_f1"]].to_string())

    best_name = min(fitted, key=lambda k: fitted[k][1])  # chosen on CV, not on the test set
    best = fitted[best_name][0]
    proba = best.predict_proba(Xte)
    pred = proba.argmax(1)
    print(f"\nselected model (best CV log loss): {best_name}")
    print("confusion matrix on test (rows = actual, cols = predicted):")
    print(pd.DataFrame(confusion_matrix(yte, pred, labels=[0, 1, 2]), index=LABELS, columns=LABELS))
    by_comp = test.assign(correct=(pred == yte)).groupby("competition")["correct"].agg(["mean", "size"])
    print("\naccuracy by competition:\n", by_comp.rename(columns={"mean": "accuracy", "size": "matches"}).round(3))

    imp = pd.Series(best.feature_importances_, index=cols).sort_values(ascending=False)
    print("\ntop 15 features:\n", imp.head(15).round(4).to_string())

    # refit the chosen configuration on all labelled data (train + test) for live predictions
    final = type(best)(**best.get_params()).fit(data[cols], data.target.values)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"model": final, "name": best_name, "features": cols}, ROOT / "models" / "model.joblib")
    (ROOT / "reports").mkdir(exist_ok=True)
    # reliability (calibration) data: predicted probability vs how often the outcome really happened
    calibration = {}
    for k, label in enumerate(LABELS):
        bins = pd.cut(proba[:, k], np.linspace(0, 1, 11))
        grp = pd.DataFrame({"pred": proba[:, k], "hit": (yte == k).astype(float), "bin": bins})
        grp = grp.groupby("bin", observed=True).agg(predicted=("pred", "mean"), actual=("hit", "mean"),
                                                     n=("hit", "size"))
        calibration[label] = grp[grp.n >= 15].round(4).to_dict(orient="records")

    report = {"selected": best_name, "train_matches": len(train), "test_matches": len(test),
              "train_period": [f"{train.date.min():%Y-%m}", f"{train.date.max():%Y-%m}"],
              "test_period": [f"{test.date.min():%Y-%m}", f"{test.date.max():%Y-%m}"],
              "test_seasons_from": args.test_from, "results": results,
              "confusion_matrix": confusion_matrix(yte, pred, labels=[0, 1, 2]).tolist(),
              "accuracy_by_competition": by_comp.rename(columns={"mean": "accuracy", "size": "matches"})
                                                .round(4).reset_index().to_dict(orient="records"),
              "calibration": calibration,
              "feature_importance": imp.round(5).to_dict()}
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(report, indent=2, default=str))
    print("\nsaved models/model.joblib and reports/metrics.json")


if __name__ == "__main__":
    main()
