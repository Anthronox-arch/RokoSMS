import joblib

from rokosms.data import load_train, split_xy
from rokosms.models import build_svc_calibrated_pipeline, DECISION_THRESHOLD

VERSION = "v1"

train_df = load_train()
X_train, y_train, _ = split_xy(train_df, mask_phone=True)

model = build_svc_calibrated_pipeline()
model.fit(X_train, y_train)

artifact = {
    "model": model,
    "threshold": DECISION_THRESHOLD,
    "version": VERSION,
    "mask_phone": True,
}

path = f"models/rokosms_{VERSION}.joblib"
joblib.dump(artifact, path)
print(f"Saved {path}")
