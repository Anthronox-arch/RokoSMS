import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_score, recall_score

from rokosms.data import load_train, split_xy
from rokosms.models import build_svc_calibrated_pipeline

N_FOLDS = 5
THRESHOLDS = np.arange(0.05, 0.96, 0.05)
MIN_PRECISION = 0.90

train_df = load_train()
X, y, groups = split_xy(train_df, mask_phone=True)

splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

all_probs = np.zeros(len(y))
for train_idx, test_idx in splitter.split(X, y, groups):
    model = build_svc_calibrated_pipeline()
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    all_probs[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]

print(f"{'Threshold':>10} {'Recall':>10} {'Precision':>10} {'OK?':>6}")
best_threshold = None
best_recall = -1
for t in THRESHOLDS:
    preds = (all_probs >= t).astype(int)
    recall = recall_score(y, preds)
    precision = precision_score(y, preds)
    ok = precision >= MIN_PRECISION
    print(f"{t:>10.2f} {recall:>10.3f} {precision:>10.3f} {'YES' if ok else 'no':>6}")
    if ok and recall > best_recall:
        best_recall = recall
        best_threshold = t

print(f"\nBest threshold meeting precision >= {MIN_PRECISION}: {best_threshold}")
print(f"  -> recall: {best_recall:.3f}")
