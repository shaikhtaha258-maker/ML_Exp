import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


# Load Dataset

df = pd.read_csv("house_price.csv")

print("First Five Records")
print(df.head())

print("\nDataset Shape")
print(df.shape)


# Features and Target

X = df.drop("Price", axis=1)
y = df["Price"]


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Standard Scaling

standard = StandardScaler()

X_train_std = standard.fit_transform(X_train)
X_test_std = standard.transform(X_test)

print("\nStandard Scaling Completed")
print("First Five Records")
print(pd.DataFrame(X_train_std).head())


# Min-Max Scaling

minmax = MinMaxScaler()

X_train_mm = minmax.fit_transform(X_train)
X_test_mm = minmax.transform(X_test)

print("\nMin-Max Scaling Completed")


# Linear Regression

linear = LinearRegression()

linear.fit(X_train_std, y_train)

y_pred_linear = linear.predict(X_test_std)

print("\nLinear Regression")
print("R2 Score :", r2_score(y_test, y_pred_linear))
print("MSE :", mean_squared_error(y_test, y_pred_linear))


# Ridge Regression

ridge = Ridge(alpha=1.0)

ridge.fit(X_train_std, y_train)

y_pred_ridge = ridge.predict(X_test_std)

print("\nRidge Regression")
print("R2 Score :", r2_score(y_test, y_pred_ridge))
print("MSE :", mean_squared_error(y_test, y_pred_ridge))


# Lasso Regression

lasso = Lasso(alpha=0.01)

lasso.fit(X_train_std, y_train)

y_pred_lasso = lasso.predict(X_test_std)

print("\nLasso Regression")
print("R2 Score :", r2_score(y_test, y_pred_lasso))
print("MSE :", mean_squared_error(y_test, y_pred_lasso))


# Comparison

comparison = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression"
    ],

    "R2 Score": [
        r2_score(y_test, y_pred_linear),
        r2_score(y_test, y_pred_ridge),
        r2_score(y_test, y_pred_lasso)
    ],

    "MSE": [
        mean_squared_error(y_test, y_pred_linear),
        mean_squared_error(y_test, y_pred_ridge),
        mean_squared_error(y_test, y_pred_lasso)
    ]

})

print("\nModel Comparison")
print(comparison)