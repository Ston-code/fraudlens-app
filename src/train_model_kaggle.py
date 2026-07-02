import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ─── 1. Load Real Dataset ─────────────────────────────────────────────────────
print("\nLoading dataset...")
df = pd.read_csv("creditcard.csv")

print(f"  Total transactions : {len(df):,}")
print(f"  Fraud cases        : {df['Class'].sum():,}")
print(f"  Fraud rate         : {df['Class'].mean()*100:.3f}%")

# ─── 2. Features & Target ─────────────────────────────────────────────────────
features = ["Amount", "Time"] + [f"V{i}" for i in range(1, 29)]
X = df[features]
y = df["Class"]

# ─── 3. Scale Amount and Time ─────────────────────────────────────────────────
X = X.copy()
amount_mean = df["Amount"].mean()
amount_std  = df["Amount"].std()
time_mean   = df["Time"].mean()
time_std    = df["Time"].std()

X["Amount"] = (X["Amount"] - amount_mean) / amount_std
X["Time"]   = (X["Time"]   - time_mean)   / time_std

# ─── 4. Train / Test Split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  Training samples   : {len(X_train):,}")
print(f"  Test samples       : {len(X_test):,}")

# ─── 5. Train Logistic Regression ────────────────────────────────────────────
print("\nTraining model... (this may take a moment)")
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
    solver="lbfgs"
)
model.fit(X_train, y_train)

# ─── 6. Evaluate Model ────────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "="*55)
print("       FRAUD DETECTION MODEL — TRAINING RESULTS")
print("="*55)
print(f"\n  AUC-ROC Score : {roc_auc_score(y_test, y_proba):.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

print("\n  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                 Predicted Legit   Predicted Fraud")
print(f"  Actual Legit   {cm[0][0]:>14,}   {cm[0][1]:>15,}")
print(f"  Actual Fraud   {cm[1][0]:>14,}   {cm[1][1]:>15,}")

# ─── 7. Print Weights for App.jsx ────────────────────────────────────────────
coef   = model.coef_[0]
interc = model.intercept_[0]

print("="*55)
print("  COPY THESE WEIGHTS INTO YOUR App.jsx")
print("="*55)
print("\nconst MODEL_WEIGHTS = {")
print(f"  intercept: {interc:.6f},")
for name, val in zip(features, coef):
    print(f"  {name}: {val:.6f},")
print("};")

print("\n" + "="*55)
print("  COPY THESE SCALING PARAMS INTO YOUR App.jsx")
print("="*55)
print(f"\nconst SCALING = {{")
print(f"  amount_mean: {amount_mean:.6f},")
print(f"  amount_std:  {amount_std:.6f},")
print(f"  time_mean:   {time_mean:.6f},")
print(f"  time_std:    {time_std:.6f},")
print(f"}};")
print("\n" + "="*55)
print("  Model training complete!")
print("="*55 + "\n")
