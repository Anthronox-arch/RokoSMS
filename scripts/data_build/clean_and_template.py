"""
clean_and_template.py

Step 1 of the RokoSMS data-cleaning pipeline.

Goal: turn the 1000-row found dataset (which is template-generated: a small
set of underlying message templates with slot values swapped in --
bank/wallet name, amount, OTP digits, phone number, transaction ID -- plus
throwaway Roman Urdu spelling-variant noise on verb endings) into:

  1. A fully deduplicated TEMPLATE table: one row per genuinely distinct
     message structure, with entity values replaced by placeholders. This
     is what should be shown to the LLM generator as seed material.
  2. A cleaned full-message table with spelling normalized and a
     `template_key` column, so we can always check later whether a new
     (real or LLM-generated) message collapses onto a template we've
     already seen.
"""

import re
import pandas as pd

COLS = ["ID", "Raw_text", "Labels", "Source_Type", "Normalized_text"]

# ---------------------------------------------------------------------------
# 1. Load + combine. The provided train/test split is a plain ID-order slice
#    (799-1000 landed in "test"), not a stratified random split, and this
#    whole file is a found/templated dataset rather than organically-collected
#    real messages -- so we treat it as one pool now, clean it, and will cut a
#    fresh stratified split at the very end once we know the real dedup size.
# ---------------------------------------------------------------------------
train = pd.read_excel("/mnt/user-data/uploads/train_phishing_dataset_800_1000.xlsx")
test = pd.read_excel("/mnt/user-data/uploads/test_phishing_dataset_200_1000.xlsx", header=None)
test.columns = COLS
df = pd.concat([train, test], ignore_index=True)

# ---------------------------------------------------------------------------
# 2. Spelling-variant normalization map.
#    This dataset's ONLY source of "diversity" within a template is throwaway
#    verb-ending spelling (krn/kren/karin/karo -> karein, h/ha -> hai, etc).
#    Collapsing these first is what lets templating find the real duplicates.
#    NOTE: this list is a starting point from what's observed in THIS data --
#    expand it as real form submissions surface new variants.
# ---------------------------------------------------------------------------
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
    r"\bacount\b": "account",  # common typo variant seen in this data
    r"e-\s+challan": "e-challan",  # collapse "e- challan" / "e-challan" spacing
}

def normalize_spelling(text: str) -> str:
    t = text.lower()
    for pattern, repl in SPELLING_VARIANTS.items():
        t = re.sub(pattern, repl, t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ---------------------------------------------------------------------------
# 3. Entity templating -- replace the things that make each row LOOK unique
#    but carry no classification signal (exact OTP digits, exact amount,
#    exact phone number, exact transaction id) with placeholders.
#    Order matters: most specific pattern first.
# ---------------------------------------------------------------------------
BRANDS = [
    "nayapay", "sadapay", "jazzcash", "easypaisa", "hbl", "ubl", "mcb",
    "meezan bank", "meezan", "allied bank", "bank alfalah", "askari bank",
    "bank al-habib", "faysal bank", "lesco", "iesco", "wasa", "psca",
    "fbr", "bisp", "benazir income support", "inaam ghar",
]

def templatize(text_norm: str) -> str:
    """
    Sentence-scoped entity classification: instead of matching one fixed
    word order ("otp ... is DIGIT"), split into sentences and, if a
    sentence mentions the relevant keyword ANYWHERE, replace any digit run
    in that sentence with the matching placeholder. This survives natural
    phrasing variation ("Use OTP {OTP} to...", "Transaction ID: {trxid}.",
    "{OTP} is your OTP...") instead of only the one word order the original
    1000-row dataset happened to always use.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text_norm)

    # Phone numbers FIRST, globally, before the sentence-scoped OTP/TRXID
    # pass -- otherwise a phone number sitting in the same sentence as the
    # word "OTP" (e.g. "...bhejein is number par: {PHONE}" right after an
    # OTP mention) gets swallowed by the OTP rule instead of recognized as
    # a phone number, since both are just "a run of digits" to a naive pass.
    sentences = [re.sub(r"\b0\d{2,4}[-\s]?\d{6,8}\b", "{PHONE}", s) for s in sentences]

    out_sentences = []
    for s in sentences:
        if re.search(r"\btrx\b|transaction\s*(id|number|ka|ki)|\bref\b|reference|shanakhti|hawala|\bid\b|pehchan|\brecord\b|len den", s, re.IGNORECASE):
            s = re.sub(r"\b\d{4,10}\b", "{TRXID}", s)
        elif re.search(r"\botp\b|verification|tasdeeq|one-time|security code|\bpassword\b", s, re.IGNORECASE):
            s = re.sub(r"\b\d{4,10}\b", "{OTP}", s)
        out_sentences.append(s)
    t = " ".join(out_sentences)

    # amounts: digits (with optional commas) immediately followed by pkr/rs,
    # OR the "1 lakh pkr" phrasing
    t = re.sub(r"\b\d[\d,]*\s*(pkr|rs\.?|rupees)\b", "{AMOUNT} \\1", t)
    t = re.sub(r"\b\d+\s*lakh\b", "{AMOUNT_WORDS} lakh", t)

    # any remaining bare 4+ digit number we haven't classified -> generic
    t = re.sub(r"\b\d{4,}\b", "{NUM}", t)

    # brand/institution names -> {BRAND}, so the classifier/LLM sees the
    # STRUCTURE is shared across banks/wallets rather than treating each
    # bank name as its own separate template
    for b in sorted(BRANDS, key=len, reverse=True):
        t = re.sub(re.escape(b), "{BRAND}", t, flags=re.IGNORECASE)

    t = re.sub(r"\s+", " ", t).strip()
    return t

df["clean_text"] = df["Raw_text"].apply(normalize_spelling)
df["template_key"] = df["clean_text"].apply(templatize)

print(f"Total rows: {len(df)}")
print(f"Unique clean_text (spelling-normalized only): {df['clean_text'].nunique()}")
print(f"Unique template_key (spelling + entities collapsed): {df['template_key'].nunique()}")

# sanity check: does every row in a template_key group share the same label?
bad = df.groupby("template_key")["Labels"].nunique()
bad = bad[bad > 1]
print(f"\nTemplate groups with INCONSISTENT labels: {len(bad)}")
if len(bad):
    print(bad.head(10))

df.to_csv("/home/claude/rokosms/full_cleaned_with_templates.csv", index=False)
print("\nSaved full_cleaned_with_templates.csv")

# ---------------------------------------------------------------------------
# 4. Collapse to ONE representative row per unique template_key.
#    Representative = the shortest raw text in the group (tends to be the
#    cleanest-spelled variant, since noisy variants tend to have odd spacing
#    or extra characters) -- ties broken by lowest ID.
# ---------------------------------------------------------------------------
df["raw_len"] = df["Raw_text"].str.len()
df_sorted = df.sort_values(["template_key", "raw_len", "ID"])
unique_templates = df_sorted.groupby("template_key", as_index=False).first()
unique_templates = unique_templates.merge(
    df.groupby("template_key").size().rename("n_occurrences"),
    on="template_key",
)
unique_templates = unique_templates[
    ["template_key", "Raw_text", "clean_text", "Labels", "Source_Type", "n_occurrences"]
].sort_values("n_occurrences", ascending=False)

unique_templates.to_csv("/home/claude/rokosms/unique_templates_seed_set.csv", index=False)
print(f"\nSaved unique_templates_seed_set.csv: {len(unique_templates)} rows")
print(f"  Label 0 (legit) templates: {(unique_templates['Labels']==0).sum()}")
print(f"  Label 1 (phishing) templates: {(unique_templates['Labels']==1).sum()}")
