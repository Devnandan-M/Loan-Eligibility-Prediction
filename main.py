import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay


df = pd.read_csv("Loan_Data.csv")

df = df.drop("Loan_ID", axis=1)
df = df.drop("Gender", axis=1)
df = df.dropna(subset=["Married","Loan_Amount_Term","Credit_History","Self_Employed","LoanAmount"])
df['Dependents'] = df['Dependents'].str.replace('+', '', regex=False)
df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce').astype('Int64')
df["Dependents"] = df["Dependents"].fillna(0)
df = pd.get_dummies(df, drop_first=True)
X = df.drop("Loan_Status_Y", axis=1)
y = df["Loan_Status_Y"]
print(df.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=100,
    stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=100)
model.fit(X_train_scaled, y_train)

print("Logistic Regression model trained successfully!")

prediction = model.predict(X_test_scaled)

precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

actual_counts = y_test.value_counts().sort_index()

predicted_counts = pd.Series(prediction).value_counts().sort_index()

labels = ["No Stroke (0)", "Stroke (1)"]
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(
    x - width / 2,
    actual_counts,
    width,
    label="Actual",
    color="skyblue",
    edgecolor="black",
)
ax.bar(
    x + width / 2,
    predicted_counts,
    width,
    label="Predicted",
    color="salmon",
    edgecolor="black",
)

ax.set_xlabel("Class Category", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title(
    "Comparison of Actual vs. Predicted Class Distributions",
    fontsize=14,
    weight="bold",
)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.tight_layout()
plt.savefig("Loan Eligibility prediction comparison.png", dpi=300, bbox_inches="tight")


# Plot the confusion matrix directly from true and predicted values
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, prediction, display_labels=["Not Eligible", "Eligible"], cmap="Blues", ax=ax
)
print(prediction)
plt.title("Confusion Matrix", fontsize=14, weight="bold")

# Optional: Save the plot to your device
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")