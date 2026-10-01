import pandas as pd
from rokosms.text_utils import normalize_spelling, templatize

master = pd.read_csv("../../data/processed/pipeline_dataset_MASTER_POOL.csv")
master["recomputed_key"] = master["text"].apply(lambda x: templatize(normalize_spelling(x)))
mismatches = master[master["recomputed_key"].str.lower() != master["template_key"].str.lower()]

print(f"Total mismatches: {len(mismatches)}")
print(mismatches[["source", "Source_Type", "text"]].to_string())
