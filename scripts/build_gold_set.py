import pandas as pd

raw = pd.read_csv("data/gold/gold_raw.csv")
raw = raw.dropna(how="all")

SCAM_TEXT_COL = "Paste the exact message text. Ensure it is Roman Urdu only, and any of your personal information (if any) has been stripped.\n\nDo not remove any URLS, names, or numbers of the sender that may be contained within.\n\nYou can submit this form again for each additional message."
LEGIT_TEXT_COL = "Paste the exact message here. Remove any information you do not want to share, if any.\n\nYou can submit this form again for each additional message."
TYPE_COL = "What type of message is this?"

rows = []
for _, row in raw.iterrows():
    is_scam = "Scam" in str(row[TYPE_COL])
    text = row[SCAM_TEXT_COL] if is_scam else row[LEGIT_TEXT_COL]
    if pd.isna(text):
        continue
    rows.append({
        "text": str(text).replace("\n", " ").strip(),
        "Labels": 1 if is_scam else 0,
    })

gold = pd.DataFrame(rows)
print(f"Gold set: {len(gold)} rows")
print(gold["Labels"].value_counts())

gold.to_csv("data/gold/gold_test.csv", index=False)
print("\nSaved data/gold/gold_test.csv")
