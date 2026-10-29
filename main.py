import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("Loan_Data.csv")

df = df.drop("Loan_ID", axis=1)
df = df.drop("Gender", axis=1)
df = df.dropna(subset=["Married","Loan_Amount_Term","Credit_History","Self_Employed","LoanAmount"])
df['Dependents'] = df['Dependents'].str.replace('+', '', regex=False)
df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce').astype('Int64')
df["Dependents"] = df["Dependents"].fillna(0)


df = pd.get_dummies(df, drop_first=True)
print(df.columns.tolist())
X = df.drop("Loan_Status_Y", axis=1)
y = df["Loan_Status_Y"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=100)

model.fit(X_train_scaled, y_train)

print("Logistic Regression model trained successfully!")
