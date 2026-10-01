import joblib

from .text_utils import mask_phones
from .explain import explain_message

from .url_check import check_url_mismatch
from .text_utils import BRANDS

_model_state = {}


def _get_model():
    if not _model_state:
        artifact = joblib.load("models/rokosms_v1.joblib")
        _model_state["model"] = artifact["model"]
        _model_state["threshold"] = artifact["threshold"]
    return _model_state


def _clean_reasons(pairs):
    seen_words = set()
    cleaned = []
    for name, value in pairs:
        text = name.split("__", 1)[1]
        words = set(text.split())
        if words & seen_words:
            continue
        seen_words |= words
        cleaned.append(text)
    return cleaned


MIN_REASON_STRENGTH = 0.02


def predict(text, top_n=3):
    state = _get_model()
    masked_text = mask_phones(text)

    proba = state["model"].predict_proba([masked_text])[0][1]
    label = 1 if proba >= state["threshold"] else 0

    reasons = explain_message(masked_text, top_n=10)
    strong_scam = [p for p in reasons["toward_scam"] if p[1] >= MIN_REASON_STRENGTH]
    strong_legit = [p for p in reasons["toward_legit"] if p[1] <= -MIN_REASON_STRENGTH]

    toward_scam = _clean_reasons(strong_scam)[:top_n]
    toward_legit = _clean_reasons(strong_legit)[:top_n]

    url_result = check_url_mismatch(text, BRANDS)

    return {
        "label": "scam" if label == 1 else "legit",
        "probability": round(float(proba), 3),
        "toward_scam": toward_scam,
        "toward_legit": toward_legit,
        "url_check": url_result
    }
