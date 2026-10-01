"""
build_final_pipeline_dataset.py

Assembles the actual dataset the model will be trained/evaluated on, from
everything produced so far:
  - REAL: the 221 deduplicated real templates (unique_templates_seed_set.csv)
    -- used AS-IS, one instance per template, since these are real observed
    messages (re-randomizing their entities would mean discarding real data
    and replacing it with fake data, which defeats the point).
  - SYNTHETIC: the 351 validated synthetic templates (synthetic_combined_ALL.csv)
    -- each filled into multiple realistic instances in code (never by an
    LLM), using the same consistent-value-per-placeholder logic fixed
    earlier (the brand-mismatch and phone/OTP-collision bugs).

Then: a full-set validation scan (not just per-batch), and a
TEMPLATE-GROUPED train/dev split -- grouped so that multiple filled
instances of the same template never end up split across train and dev,
which would leak information and inflate the dev score.

IMPORTANT CAVEAT baked into the output: this dev split is a DEVELOPMENT
split for iterating on the model now. It is NOT the final real-only gold
test set -- that still must come later from actual Google Form submissions,
per the earlier decision that synthetic/found data never enters the true
held-out evaluation set.
"""

import random
import re
import pandas as pd
from rokosms.text_utils import normalize_spelling, templatize

random.seed(42)

BRANDS = ["HBL", "UBL", "MCB", "Meezan Bank", "Allied Bank", "Bank Alfalah",
          "Askari Bank", "Faysal Bank", "Easypaisa", "JazzCash", "NayaPay", "SadaPay"]

def fake_phone():
    return f"03{random.randint(0,9)}{random.randint(1000000,9999999)}"
def fake_otp():
    return str(random.randint(100000, 99999999))
def fake_trxid():
    return str(random.randint(10000000, 99999999))
def fake_amount():
    return f"{random.choice([500,1000,2500,5000,7500,10000,12500,25000,50000,75000]):,}"
def fake_num():
    return str(random.randint(1000, 999999))

def fill_template(template: str) -> str:
    replacements = {
        "{BRAND}": random.choice(BRANDS), "{PHONE}": fake_phone(), "{OTP}": fake_otp(),
        "{TRXID}": fake_trxid(), "{AMOUNT}": fake_amount(), "{NUM}": fake_num(),
    }
    t = template
    for placeholder, value in replacements.items():
        t = t.replace(placeholder, value)
    return t

# ---------------------------------------------------------------------------
# 1. Load REAL data -- one instance per real template, as-is.
# ---------------------------------------------------------------------------
real = pd.read_csv("../../data/raw/unique_templates_seed_set.csv")[["Labels", "Source_Type", "Raw_text", "template_key"]]
real = real.rename(columns={"Raw_text": "text"})
real["source"] = "real"
print(f"Real messages: {len(real)}  (phishing={ (real.Labels==1).sum() }, legit={ (real.Labels==0).sum() })")

# ---------------------------------------------------------------------------
# 2. Load SYNTHETIC templates (already validated, comma-delimited, deduped
#    against everything, from the previous combine-and-scan step) and fill
#    each into FILLS_PER_TEMPLATE realistic instances.
# ---------------------------------------------------------------------------
FILLS_PER_TEMPLATE = 4
synth_templates = pd.read_csv("../../data/raw/synthetic_combined_ALL.csv")
synth_templates["template_key"] = synth_templates["raw_template"].apply(
    lambda x: templatize(normalize_spelling(x))
)

synth_rows = []
for _, r in synth_templates.iterrows():
    seen_for_this_template = set()
    attempts = 0
    while len(seen_for_this_template) < FILLS_PER_TEMPLATE and attempts < FILLS_PER_TEMPLATE * 5:
        filled = fill_template(r["raw_template"])
        attempts += 1
        if filled in seen_for_this_template:
            continue
        seen_for_this_template.add(filled)
        synth_rows.append({
            "Labels": r["Labels"], "Source_Type": r["Source_Type"], "text": filled,
            "template_key": r["template_key"], "source": "synthetic",
        })
synthetic = pd.DataFrame(synth_rows)
print(f"Synthetic messages: {len(synthetic)}  (phishing={ (synthetic.Labels==1).sum() }, legit={ (synthetic.Labels==0).sum() })"
      f"  [from {len(synth_templates)} templates x {FILLS_PER_TEMPLATE} fills]")

# ---------------------------------------------------------------------------
# 3. Combine into the master pool
# ---------------------------------------------------------------------------
master = pd.concat([real, synthetic], ignore_index=True)
print(f"\nMASTER POOL: {len(master)} total messages")
print(master.groupby(["source", "Labels"]).size())

# ---------------------------------------------------------------------------
# 4. FULL-SET validation scan -- on the final filled master pool, not just
#    the templates. Checks things that only exist post-filling.
# ---------------------------------------------------------------------------
print("\n=== FULL-SET VALIDATION SCAN ===")

# 4a. No literal placeholder tokens should remain anywhere
leftover_tokens = master["text"].str.contains(r"\{[A-Z_]+\}", regex=True).sum()
print(f"Rows with an UNFILLED placeholder token still in the text: {leftover_tokens}")

# 4b. Exact duplicate final text, across the ENTIRE pool (real + synthetic
#     together) -- a random fill of one template coincidentally matching
#     another template's fill, or matching a real message verbatim
exact_dupes = master["text"].duplicated().sum()
print(f"Exact duplicate final message text across the WHOLE pool: {exact_dupes}")

# 4c. Label/category consistency
mismatch = master[
    (master["Source_Type"].str.startswith("Phishing") & (master["Labels"] != 1)) |
    (~master["Source_Type"].str.startswith("Phishing") & (master["Labels"] != 0))
]
print(f"Label/category mismatches: {len(mismatch)}")

# 4d. Round-trip: every filled message should still collapse back to ITS
#     OWN stored template_key (case-insensitive) -- catches any residual
#     templatize() blind spot on this specific batch of fills
master["recomputed_key"] = master["text"].apply(lambda x: templatize(normalize_spelling(x)))
roundtrip_mismatch = master["recomputed_key"].str.lower() != master["template_key"].str.lower()
print(f"Round-trip template mismatches: {roundtrip_mismatch.sum()}")

# 4e. Null / empty check
print(f"Null text rows: {master['text'].isna().sum()}")

if leftover_tokens or exact_dupes or len(mismatch) or roundtrip_mismatch.sum():
    print("\n*** ISSUES FOUND -- see details above before proceeding ***")
else:
    print("\nAll checks clean.")

master = master.drop(columns=["recomputed_key"])
master.to_csv("../../data/processed/pipeline_dataset_MASTER_POOL.csv", index=False)
print(f"\nSaved pipeline_dataset_MASTER_POOL.csv: {len(master)} rows")
