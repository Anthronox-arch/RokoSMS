import pandas as pd

train = pd.read_csv("data/processed/pipeline_train.csv")
dev = pd.read_csv("data/processed/pipeline_dev.csv")

for name, df in [("TRAIN", train), ("DEV", dev)]:
    print(f"\n===== {name}: {len(df)} rows =====")
    print("\nLabels:")
    print(df["Labels"].value_counts())
    print("\nSource x Labels:")
    print(df.groupby(["source", "Labels"]).size())
    print("\nRows per Source_Type:")
    print(df["Source_Type"].value_counts())
    lengths = df["text"].str.len()
    print("\nMessage length (characters):")
    print(lengths.describe())

print("\n===== 2 RANDOM EXAMPLES PER CATEGORY (train) =====")
for cat, group in train.groupby("Source_Type"):
    print(f"\n--- {cat} ---")
    for text in group["text"].sample(min(2, len(group)), random_state=1):
        print("  *", text)
