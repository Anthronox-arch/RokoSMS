import pandas as pd

from rokosms.predict import predict

gold = pd.read_csv("data/gold/gold_test.csv")

results = []
for _, row in gold.iterrows():
    pred = predict(row["text"])
    pred_label = 1 if pred["label"] == "scam" else 0
    results.append({
        "text": row["text"],
        "true_label": row["Labels"],
        "pred_label": pred_label,
        "probability": pred["probability"],
        "correct": pred_label == row["Labels"],
    })

results_df = pd.DataFrame(results)

print("=== FULL RESULTS ===")
for _, r in results_df.iterrows():
    mark = "CORRECT" if r["correct"] else "WRONG"
    print(f"[{mark}] true={r['true_label']} pred={r['pred_label']} prob={r['probability']:.3f} | {r['text'][:80]}")

n_scam = (results_df["true_label"] == 1).sum()
n_scam_correct = ((results_df["true_label"] == 1) & (results_df["correct"])).sum()
recall = n_scam_correct / n_scam if n_scam > 0 else float("nan")

n_legit = (results_df["true_label"] == 0).sum()
n_legit_correct = ((results_df["true_label"] == 0) & (results_df["correct"])).sum()

print(f"\n=== SUMMARY ===")
print(f"Scam messages: {n_scam}, correctly caught: {n_scam_correct}, recall: {recall:.3f}")
print(f"Legit messages: {n_legit}, correctly identified: {n_legit_correct} (too few to compute meaningful precision)")

results_df.to_csv("reports/gold_eval_results.csv", index=False)
print("\nSaved reports/gold_eval_results.csv")
