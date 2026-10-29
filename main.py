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

print("Dataset shape after encoding:", df.shape)
