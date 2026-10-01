import pandas as pd

train = pd.read_csv("data/processed/pipeline_train.csv")
dev = pd.read_csv("data/processed/pipeline_dev.csv")

# Check 1: exact text overlap between train and dev
overlap = set(train["text"]) & set(dev["text"])
print(f"Exact text rows shared between train and dev: {len(overlap)}")
if overlap:
    print("Example overlapping text:", list(overlap)[0])

# Check 2: how many unique templates does each dev category have?
print("\nUnique template_key count per Source_Type in DEV:")
print(dev.groupby("Source_Type")["template_key"].nunique())

# Check 3: real-only rows in dev, so we can score on just those later
real_dev = dev[dev["source"] == "real"]
print(f"\nReal-only dev rows: {len(real_dev)} (phishing={ (real_dev.Labels==1).sum() }, legit={ (real_dev.Labels==0).sum() })")
