import joblib

from rokosms.text_utils import normalize_spelling, templatize, mask_phones
from rokosms.predict import predict
from rokosms.url_check import check_url_mismatch
from rokosms.text_utils import BRANDS


def test_templatize_otp():
    text = normalize_spelling("Your OTP is 123456")
    result = templatize(text)
    assert result == "your otp is {OTP}"


def test_mask_phones():
    result = mask_phones("Call now at 0348831655 to claim")
    assert "0348831655" not in result
    assert result == "Call now at to claim"


def test_url_extraction_catches_shortened_links():
    from rokosms.url_check import extract_urls
    urls = extract_urls("Verify now: bit.ly/482910")
    assert len(urls) == 1
    assert "bit.ly" in urls[0]


def test_predict_shape():
    result = predict("Your salary of 45,000 PKR has been credited to your UBL account.")
    assert result["label"] in ["scam", "legit"]
    assert 0.0 <= result["probability"] <= 1.0
    assert isinstance(result["toward_scam"], list)
    assert isinstance(result["toward_legit"], list)
    assert "url_check" in result


def test_predict_catches_obvious_phishing():
    result = predict("URGENT: Your HBL account has been suspended. Verify now: bit.ly/482910")
    assert result["label"] == "scam"


def test_url_mismatch_catches_lookalike_domain():
    result = check_url_mismatch(
        "Your HBL account is suspended. Verify at hbl.com.fake-login.info",
        BRANDS,
    )
    assert result["checked"] is True
    assert result["mismatch"] is True


def test_url_mismatch_passes_genuine_domain():
    result = check_url_mismatch(
        "Your HBL account balance is low. Visit hbl.com for more info.",
        BRANDS,
    )
    assert result["checked"] is True
    assert result["mismatch"] is False


def test_model_save_load_roundtrip():
    artifact = joblib.load("models/rokosms_v1.joblib")
    model = artifact["model"]
    threshold = artifact["threshold"]

    proba = model.predict_proba(["Your account has been suspended, verify now"])[0][1]
    assert 0.0 <= proba <= 1.0
    assert isinstance(threshold, float)
