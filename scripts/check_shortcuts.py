import re
import pandas as pd

train = pd.read_csv("data/processed/pipeline_train.csv")

phone = re.compile(r"\b0\d{9,10}\b")
url = re.compile(r"https?://|www\.|\.com|\.pk|bit\.ly", re.IGNORECASE)

train["has_phone"] = train["text"].str.contains(phone)
train["has_url"] = train["text"].str.contains(url)

print(train.groupby(["source", "Labels"])[["has_phone", "has_url"]].mean().round(2))

lengths = train["text"].str.findall(r"\b0\d{9,10}\b").explode().dropna().str.len()
print("\nPhone-like number lengths (digits):")
print(lengths.value_counts())
