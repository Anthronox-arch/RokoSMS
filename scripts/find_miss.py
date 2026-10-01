from rokosms.data import load_train, load_dev, split_xy
from rokosms.models import MODELS

train_df = load_train()
dev_df = load_dev()

X_train, y_train, _ = split_xy(train_df, mask_phone=True)
X_dev, y_dev, _ = split_xy(dev_df, mask_phone=True)

model = MODELS["linear_svc"]()
model.fit(X_train, y_train)
y_pred = model.predict(X_dev)

dev_df = dev_df.reset_index(drop=True)
for i in range(len(dev_df)):
    if y_pred[i] != y_dev.iloc[i]:
        print("TRUE LABEL:", y_dev.iloc[i], "| PREDICTED:", y_pred[i])
        print("SOURCE:", dev_df.loc[i, "source"], "| CATEGORY:", dev_df.loc[i, "Source_Type"])
        print("TEXT (phone masked):", X_dev.iloc[i])
        print()
