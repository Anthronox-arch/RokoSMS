from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from .features import build_features

DECISION_THRESHOLD = 0.30


def build_svc_pipeline():
    return Pipeline([
        ("features", build_features()),
        ("clf", LinearSVC(class_weight="balanced", random_state=42)),
    ])

def build_svc_calibrated_pipeline():
    return Pipeline([
        ("features", build_features()),
        ("clf", CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=42),
            method="sigmoid",
            cv=5,
        )),
    ])

def build_xgb_pipeline():
    return Pipeline([
        ("features", build_features()),
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
        )),
    ])


MODELS = {
    "linear_svc": build_svc_pipeline,
    "xgboost": build_xgb_pipeline,
}
