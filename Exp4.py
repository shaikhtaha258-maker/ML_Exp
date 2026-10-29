import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# Load Dataset

df = pd.read_csv("Housing.csv")

print("First Five Records")
print(df.head())

print("\nDataset Shape")
print(df.shape)


# Simple Linear Regression

X = df[["Area"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

simple_model = LinearRegression()

simple_model.fit(X_train, y_train)

y_pred_simple = simple_model.predict(X_test)


print("\nSimple Linear Regression")

print("MAE :", mean_absolute_error(y_test, y_pred_simple))
print("MSE :", mean_squared_error(y_test, y_pred_simple))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred_simple)))
print("R2 Score :", r2_score(y_test, y_pred_simple))


# Multiple Linear Regression

X = df[
    [
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Stories",
        "Parking",
        "Age"
    ]
]

y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

multiple_model = LinearRegression()

multiple_model.fit(X_train, y_train)

y_pred_multiple = multiple_model.predict(X_test)


print("\nMultiple Linear Regression")

print("MAE :", mean_absolute_error(y_test, y_pred_multiple))
print("MSE :", mean_squared_error(y_test, y_pred_multiple))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred_multiple)))
print("R2 Score :", r2_score(y_test, y_pred_multiple))