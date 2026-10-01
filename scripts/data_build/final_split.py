"""
final_split.py

Template-GROUPED train/dev split. Critical: multiple filled instances of
the SAME template must never be split across train and dev -- otherwise
the model could see 3 near-identical instances of a template in train and
be evaluated on the 4th, which inflates the dev score without meaning
anything about real generalization.

CAVEAT (kept explicit in the output): this is a DEVELOPMENT split for
iterating on the model now. It is NOT the final real-only gold evaluation
set -- that must still come from actual Google Form submissions later,
since neither the "found" dataset nor any synthetic data should ever be
the true measure of real-world performance.
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

master = pd.read_csv("../../data/processed/pipeline_dataset_MASTER_POOL.csv")

# Split at the TEMPLATE level (unique template_key), not the row level.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, dev_idx = next(splitter.split(master, groups=master["template_key"]))

train = master.iloc[train_idx].reset_index(drop=True)
dev = master.iloc[dev_idx].reset_index(drop=True)

# Sanity check: zero template overlap between the two splits
overlap = set(train["template_key"]) & set(dev["template_key"])
print(f"Template overlap between train/dev (must be 0): {len(overlap)}")

print(f"\nTrain: {len(train)} rows")
print(train.groupby(["source", "Labels"]).size())
print(f"\nDev: {len(dev)} rows")
print(dev.groupby(["source", "Labels"]).size())

train.to_csv("../../data/processed/pipeline_train.csv", index=False)
dev.to_csv("../../data/processed/pipeline_dev.csv", index=False)
print("\nSaved pipeline_train.csv and pipeline_dev.csv")
