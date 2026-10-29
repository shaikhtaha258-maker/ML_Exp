import warnings
warnings.filterwarnings("ignore")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load Titanic dataset
df = pd.read_csv("titanic.csv")

# Select features and target
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].copy()
y = df["Survived"]

# Convert Sex into numeric values
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})

# Handle  missing values
X["Age"] = X["Age"].fillna(X["Age"].median())
X["Fare"] = X["Fare"].fillna(X["Fare"].median())

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Naïve Bayes

nb = GaussianNB()

# Train the model
nb.fit(X_train, y_train)

# Predict test data
y_pred_nb = nb.predict(X_test)

# Calculate metrics
nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_precision = precision_score(y_test, y_pred_nb)
nb_recall = recall_score(y_test, y_pred_nb)
nb_f1 = f1_score(y_test, y_pred_nb)

print(" Naïve Bayes ")
print("Accuracy :", nb_accuracy)
print("Precision:", nb_precision)
print("Recall   :", nb_recall)
print("F1-Score :", nb_f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))



# K-Nearest Neighbour (KNN)

knn = KNeighborsClassifier(n_neighbors=4)

# Train the model
knn.fit(X_train, y_train)

# Predict test data
y_pred_knn = knn.predict(X_test)

# Calculate metrics
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_precision = precision_score(y_test, y_pred_knn)
knn_recall = recall_score(y_test, y_pred_knn)
knn_f1 = f1_score(y_test, y_pred_knn)

print("\n KNN ")
print("Accuracy :", knn_accuracy)
print("Precision:", knn_precision)
print("Recall   :", knn_recall)
print("F1-Score :", knn_f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))



# Comparison of Both Models

comparison = pd.DataFrame({
    "Model": ["Naïve Bayes", "KNN"],
    "Accuracy": [nb_accuracy, knn_accuracy],
    "Precision": [nb_precision, knn_precision],
    "Recall": [nb_recall, knn_recall],
    "F1-Score": [nb_f1, knn_f1]
})

print("\n Model Comparison ")
print(comparison.to_string(index=False))

