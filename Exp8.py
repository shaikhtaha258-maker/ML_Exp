# Experiment: SVM Classification using Iris Dataset

# Step 1: Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Step 2: Load Iris Dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Dataset Shape:", X.shape)
print("Feature Names:", iris.feature_names)
print("Target Names:", iris.target_names)

# Step 3: Display first five records
df = pd.DataFrame(X, columns=iris.feature_names)
df["Target"] = y

print("\nFirst Five Records:")
print(df.head())

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# Step 5: Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# MODEL 1: SVM with Linear Kernel
# =========================================================

linear_svm = SVC(
    kernel="linear",
    C=1.0,
    random_state=42
)

# Train model
linear_svm.fit(X_train_scaled, y_train)

# Prediction
y_pred_linear = linear_svm.predict(X_test_scaled)

# Evaluation
print("\n================================")
print("SVM WITH LINEAR KERNEL")
print("================================")

print("Accuracy:",
      accuracy_score(y_test, y_pred_linear))

print("Precision:",
      precision_score(
          y_test,
          y_pred_linear,
          average="weighted"
      ))

print("Recall:",
      recall_score(
          y_test,
          y_pred_linear,
          average="weighted"
      ))

print("F1-Score:",
      f1_score(
          y_test,
          y_pred_linear,
          average="weighted"
      ))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_linear,
        target_names=iris.target_names
    )
)

# Confusion Matrix
cm_linear = confusion_matrix(
    y_test,
    y_pred_linear
)

print("Confusion Matrix:")
print(cm_linear)

ConfusionMatrixDisplay(
    confusion_matrix=cm_linear,
    display_labels=iris.target_names
).plot()

plt.title("Confusion Matrix - Linear SVM")
plt.show()


# =========================================================
# MODEL 2: SVM with RBF Kernel
# =========================================================

rbf_svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    random_state=42
)

# Train model
rbf_svm.fit(X_train_scaled, y_train)

# Prediction
y_pred_rbf = rbf_svm.predict(X_test_scaled)

# Evaluation
print("\n================================")
print("SVM WITH RBF KERNEL")
print("================================")

print("Accuracy:",
      accuracy_score(y_test, y_pred_rbf))

print("Precision:",
      precision_score(
          y_test,
          y_pred_rbf,
          average="weighted"
      ))

print("Recall:",
      recall_score(
          y_test,
          y_pred_rbf,
          average="weighted"
      ))

print("F1-Score:",
      f1_score(
          y_test,
          y_pred_rbf,
          average="weighted"
      ))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_rbf,
        target_names=iris.target_names
    )
)

# Confusion Matrix
cm_rbf = confusion_matrix(
    y_test,
    y_pred_rbf
)

print("Confusion Matrix:")
print(cm_rbf)

ConfusionMatrixDisplay(
    confusion_matrix=cm_rbf,
    display_labels=iris.target_names
).plot()

plt.title("Confusion Matrix - RBF SVM")
plt.show()


# =========================================================
# MODEL COMPARISON
# =========================================================

linear_accuracy = accuracy_score(
    y_test,
    y_pred_linear
)

rbf_accuracy = accuracy_score(
    y_test,
    y_pred_rbf
)

print("\n================================")
print("MODEL COMPARISON")
print("================================")

print("Linear SVM Accuracy:", linear_accuracy)
print("RBF SVM Accuracy   :", rbf_accuracy)

if linear_accuracy > rbf_accuracy:
    print("\nLinear SVM performs better.")
elif rbf_accuracy > linear_accuracy:
    print("\nRBF SVM performs better.")
else:
    print("\nBoth models have the same accuracy.")