import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)

# 1️⃣ Load dataset
data = pd.read_csv(r"C:\Users\harsh\python-projects\Logistic-Regression-Classification\data\data.csv")  # apne csv ka naam use karo

# Drop useless columns
data = data.drop(columns=["id", "Unnamed: 32"], errors="ignore")

print("Dataset shape:", data.shape)
print("Columns:", data.columns)

# Target encode (M=1, B=0)
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

# Features (sab except target)
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

# 2️⃣ Train-test split + standardization
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3️⃣ Logistic Regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, y_train)

# 4️⃣ Evaluation
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(r"outputs/confusion_matrix.png")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(r"outputs/roc_curve.png")
plt.show()

print(f"ROC-AUC Score: {auc:.3f}")

# 5️⃣ Threshold tuning example
threshold = 0.6
y_pred_custom = (y_prob >= threshold).astype(int)
cm_custom = confusion_matrix(y_test, y_pred_custom)
print(f"\nConfusion Matrix at threshold={threshold}:\n", cm_custom)
