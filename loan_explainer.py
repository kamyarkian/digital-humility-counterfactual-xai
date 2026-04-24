import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Reproducibility
np.random.seed(42)
N_SAMPLES = 500

# Feature generation
income: np.ndarray = np.random.randint(30, 120, N_SAMPLES)
tenure: np.ndarray = np.random.randint(0, 15, N_SAMPLES)
credit_score: np.ndarray = np.random.randint(550, 800, N_SAMPLES)

# Rule-based labels: approved if all three thresholds are met
# This keeps the dataset auditable — you can verify any counterfactual by hand.
approved: np.ndarray = (
    (income >= 50) & (tenure >= 2) & (credit_score >= 650)
).astype(int)

df = pd.DataFrame({
    "income_k":      income,
    "tenure_years":  tenure,
    "credit_score":  credit_score,
    "approved":      approved,
})
FEATURE_COLS = ["income_k", "tenure_years", "credit_score"]
TARGET_COL   = "approved"

X = df[FEATURE_COLS]
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Model Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2%}")
# → Model Accuracy: 100.00%
# Note: Perfect accuracy is expected here — our labels are deterministic rules.
# This makes the tutorial auditable, not unrealistic.

# --- Kian's loan application ---
kian_profile = pd.DataFrame([{
    "income_k":     42,   # $42,000/year
    "tenure_years":  1,   # 1 year at current employer
    "credit_score": 610,  # Below the 650 threshold
}])

prediction = model.predict(kian_profile)
verdict    = "APPROVED" if prediction[0] == 1 else "DENIED"

print(f"\nPrediction for Kian: {prediction} ({verdict})")
# → Prediction for Kian: [0] (DENIED)