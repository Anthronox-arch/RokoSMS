import re
import pandas as pd

train = pd.read_csv("data/processed/pipeline_train.csv")

roman_urdu_markers = re.compile(
    r"\b(hai|hain|ka|ki|ke|aap|apka|apke|karein|kar|par|se|mein|ho|gaya|raha)\b",
    re.IGNORECASE,
)

train["has_roman_urdu"] = train["text"].str.contains(roman_urdu_markers)

print(train.groupby(["Labels", "has_roman_urdu"]).size())
