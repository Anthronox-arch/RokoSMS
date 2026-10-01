import re

SPELLING_VARIANTS = {
    r"\bkrn\b": "karein", r"\bkren\b": "karein", r"\bkarin\b": "karein",
    r"\bkro\b": "karein", r"\bkr len\b": "kar lein", r"\bkr lein\b": "kar lein",
    r"\bkarein\b": "karein",
    r"\bgaya h\b": "gaya hai", r"\bgaya ha\b": "gaya hai",
    r"\bhoa h\b": "hoa hai", r"\bhoa ha\b": "hoa hai",
    r"\bhua h\b": "hua hai", r"\bhua ha\b": "hua hai",
    r"\bhn\b": "hain", r"\bhen\b": "hain",
    r"\bkarne\b": "karne", r"\bkarnay\b": "karne",
    r"\bhaii\b": "hai",
    r"\bacount\b": "account",
    r"e-\s+challan": "e-challan",
}

BRANDS = [
    "nayapay", "sadapay", "jazzcash", "easypaisa", "hbl", "ubl", "mcb",
    "meezan bank", "meezan", "allied bank", "bank alfalah", "askari bank",
    "bank al-habib", "faysal bank", "lesco", "iesco", "wasa", "psca",
    "fbr", "bisp", "benazir income support", "inaam ghar",
]

PHONE_PATTERN = r"\b0\d{2,4}[-\s]?\d{6,8}\b"


def normalize_spelling(text: str) -> str:
    t = text.lower()
    for pattern, repl in SPELLING_VARIANTS.items():
        t = re.sub(pattern, repl, t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def templatize(text_norm: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text_norm)

    sentences = [re.sub(r"\b0\d{2,4}[-\s]?\d{6,8}\b", "{PHONE}", s) for s in sentences]

    out_sentences = []
    for s in sentences:
        if re.search(r"\btrx\b|transaction\s*(id|number|ka|ki)|\bref\b|reference|shanakhti|hawala|\bid\b|pehchan|\brecord\b|len den", s, re.IGNORECASE):
            s = re.sub(r"\b\d{4,10}\b", "{TRXID}", s)
        elif re.search(r"\botp\b|verification|tasdeeq|one-time|security code|\bpassword\b", s, re.IGNORECASE):
            s = re.sub(r"\b\d{4,10}\b", "{OTP}", s)
        out_sentences.append(s)
    t = " ".join(out_sentences)

    t = re.sub(r"\b\d[\d,]*\s*(pkr|rs\.?|rupees)\b", "{AMOUNT} \\1", t)
    t = re.sub(r"\b\d+\s*lakh\b", "{AMOUNT_WORDS} lakh", t)

    t = re.sub(r"\b\d{4,}\b", "{NUM}", t)

    for b in sorted(BRANDS, key=len, reverse=True):
        t = re.sub(re.escape(b), "{BRAND}", t, flags=re.IGNORECASE)

    t = re.sub(r"\s+", " ", t).strip()
    return t

def mask_phones(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(PHONE_PATTERN, "", text)).strip()
