import shap
from sklearn.svm import LinearSVC

from .data import load_train, split_xy
from .features import build_features

_state = {}


def _get_state():
    if not _state:
        train_df = load_train()
        X_text, y, _ = split_xy(train_df, mask_phone=True)

        features = build_features()
        X_matrix = features.fit_transform(X_text)

        clf = LinearSVC(class_weight="balanced", random_state=42)
        clf.fit(X_matrix, y)

        explainer = shap.LinearExplainer(clf, X_matrix)
        feature_names = features.get_feature_names_out()

        _state["features"] = features
        _state["explainer"] = explainer
        _state["feature_names"] = feature_names
    return _state


def explain_message(text, top_n=5):
    state = _get_state()
    X_matrix = state["features"].transform([text])
    shap_values = state["explainer"].shap_values(X_matrix)[0]

    present = X_matrix.toarray()[0] != 0

    pairs = [
        (name, value)
        for name, value, is_present in zip(state["feature_names"], shap_values, present)
        if is_present
    ]

    toward_scam = [p for p in pairs if p[1] > 0]
    toward_legit = [p for p in pairs if p[1] < 0]

    toward_scam.sort(key=lambda p: p[1], reverse=True)
    toward_legit.sort(key=lambda p: p[1])

    return {
        "toward_scam": toward_scam[:top_n],
        "toward_legit": toward_legit[:top_n],
    }
