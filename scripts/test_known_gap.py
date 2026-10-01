from rokosms.data import load_train, split_xy
from rokosms.models import MODELS

train_df = load_train()
X_train, y_train, _ = split_xy(train_df, mask_phone=True)

test_messages = [
    "ALERT: Meezan Bank card deactivation scheduled. Prevent deactivation: bit.ly/396900",
    "URGENT: Your HBL account has been suspended due to unusual activity. Verify now to restore access: bit.ly/482910",
    "Your UBL debit card will be suspended today due to a failed verification. Reactivate here: bit.ly/551203",
]

model = MODELS["linear_svc"]()
model.fit(X_train, y_train)
predictions = model.predict(test_messages)

for text, pred in zip(test_messages, predictions):
    print(f"Predicted: {pred} (1=phishing, 0=legit) | {text}")
