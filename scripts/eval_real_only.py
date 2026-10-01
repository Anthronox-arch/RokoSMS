import pandas as pd

from rokosms.data import load_train, load_dev, split_xy
from rokosms.models import MODELS
from scripts.train import get_scores  # reuse the scoring function

train_df = load_train()
dev_df = load_dev()
real_dev = dev_df[dev_df["source"] == "real"]

X_train, y_train, _ = split_xy(train_df)
X_real_dev, y_real_dev, _ = split_xy(real_dev)

for name, build_model in MODELS.items():
    model = build_model()
    model.fit(X_train, y_train)
    scores = get_scores(model, X_real_dev, y_real_dev)
    print(f"\n{name} on REAL-ONLY dev ({len(real_dev)} rows):")
    print(scores)
