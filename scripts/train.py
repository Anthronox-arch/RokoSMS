import json

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    recall_score, precision_score, f1_score, fbeta_score,
    average_precision_score, confusion_matrix,
)

from rokosms.data import load_train, load_dev, split_xy
from rokosms.models import MODELS

N_FOLDS = 5


def score_one(y_true, y_pred, y_score):
    return {
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "f2": fbeta_score(y_true, y_pred, beta=2),
        "pr_auc": average_precision_score(y_true, y_score),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def get_scores(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model.named_steps["clf"], "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)
    return score_one(y_test, y_pred, y_score)


def run_cv(build_model, X, y, groups):
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_scores = []
    for train_idx, test_idx in splitter.split(X, y, groups):
        model = build_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_scores.append(get_scores(model, X.iloc[test_idx], y.iloc[test_idx]))

    averaged = {}
    for key in ["recall", "precision", "f1", "f2", "pr_auc"]:
        averaged[key] = float(np.mean([s[key] for s in fold_scores]))
    return averaged


def main():
    train_df = load_train()
    dev_df = load_dev()

    results = {}

    for mask_phone in [False, True]:
        tag = "phone_masked" if mask_phone else "normal"
        X_train, y_train, groups_train = split_xy(train_df, mask_phone=mask_phone)
        X_dev, y_dev, _ = split_xy(dev_df, mask_phone=mask_phone)

        for model_name, build_model in MODELS.items():
            print(f"\n=== {model_name} ({tag}) ===")

            cv_scores = run_cv(build_model, X_train, y_train, groups_train)
            print("CV (train, 5-fold):", cv_scores)

            final_model = build_model()
            final_model.fit(X_train, y_train)
            dev_scores = get_scores(final_model, X_dev, y_dev)
            print("Dev:", dev_scores)

            results[f"{model_name}__{tag}"] = {
                "cv_train": cv_scores,
                "dev": dev_scores,
            }

    with open("reports/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved reports/baseline_results.json")


if __name__ == "__main__":
    main()
