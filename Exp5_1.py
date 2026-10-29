import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# Load dataset
df = pd.read_csv("Titanic.csv")

print("Dataset Shape:", df.shape)

print("\nFirst Five Records:")
print(df.head())


# Select features and target
X = df[["Age", "Sex", "Pclass", "Fare"]].copy()
y = df["Survived"]


# Convert Sex into numerical values
X.loc[:, "Sex"] = X["Sex"].map({"male": 0, "female": 1})


# Handle missing Age values
X.loc[:, "Age"] = X["Age"].fillna(X["Age"].mean())


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)


# Train model
model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)


# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))