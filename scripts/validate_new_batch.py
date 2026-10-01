import pandas as pd

from rokosms.text_utils import normalize_spelling, templatize

NEW_BATCH_PATH = "data/raw/english_phishing_batch.csv"
REAL_PATH = "data/raw/unique_templates_seed_set.csv"
SYNTHETIC_PATH = "data/raw/synthetic_combined_ALL.csv"

new_batch = pd.read_csv(NEW_BATCH_PATH)
real = pd.read_csv(REAL_PATH)
existing_synthetic = pd.read_csv(SYNTHETIC_PATH)

new_batch["template_key"] = new_batch["raw_template"].apply(
    lambda x: templatize(normalize_spelling(x))
)
real["template_key"] = real["Raw_text"].apply(
    lambda x: templatize(normalize_spelling(x))
)
existing_synthetic["template_key"] = existing_synthetic["raw_template"].apply(
    lambda x: templatize(normalize_spelling(x))
)

print(f"New batch: {len(new_batch)} templates")

# Check 1: duplicates inside the new batch itself
dupes_inside = new_batch["template_key"].duplicated().sum()
print(f"Duplicates inside new batch: {dupes_inside}")

# Check 2: collision against real templates
collide_real = new_batch["template_key"].isin(real["template_key"]).sum()
print(f"Collisions with REAL templates: {collide_real}")

# Check 3: collision against existing synthetic templates
collide_synth = new_batch["template_key"].isin(existing_synthetic["template_key"]).sum()
print(f"Collisions with EXISTING synthetic templates: {collide_synth}")

# Check 4: round-trip -- does each template collapse back to its own key?
recomputed = new_batch["raw_template"].apply(lambda x: templatize(normalize_spelling(x)))
mismatch = (recomputed != new_batch["template_key"]).sum()
print(f"Round-trip mismatches: {mismatch}")

# Check 5: label/category consistency
bad_label = new_batch[
    (new_batch["Source_Type"].str.startswith("Phishing") & (new_batch["Labels"] != 1))
]
print(f"Label/category mismatches: {len(bad_label)}")

# Check 6: leftover unfilled tokens that shouldn't be there (typos in placeholders)
import re
bad_tokens = new_batch["raw_template"].apply(
    lambda x: bool(re.search(r"\{[A-Z_]+\}", x)) and
    not all(tok in ["{BRAND}", "{AMOUNT}", "{OTP}", "{PHONE}", "{TRXID}", "{NUM}"]
            for tok in re.findall(r"\{[A-Z_]+\}", x))
)
print(f"Templates with unexpected placeholder names: {bad_tokens.sum()}")

if dupes_inside or collide_real or collide_synth or mismatch or len(bad_label) or bad_tokens.sum():
    print("\n*** ISSUES FOUND -- do NOT merge yet. See details above. ***")
else:
    print("\nAll checks clean. Safe to merge.")
    merged = pd.concat([existing_synthetic, new_batch[["Labels", "Source_Type", "raw_template"]]], ignore_index=True)
    merged.to_csv(SYNTHETIC_PATH, index=False)
    print(f"Merged. synthetic_combined_ALL.csv now has {len(merged)} templates.")
