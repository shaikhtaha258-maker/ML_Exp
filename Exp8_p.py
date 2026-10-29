import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay

)

wine = load_wine()
X = wine.data
y = wine.target

print("Dataset Shape:",X.shape)
print("Feature Names:", wine.feature_names)
print("Target Names:", wine.target_names)

df = pd.DataFrame(X, columns=wine.feature_names)
df["Target"] = y

print("\nFirst Five Records:")
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linear_svm = SVC(
    kernel="linear",
    C=1.0,
    random_state=42
)

# Train model
linear_svm.fit(X_train_scaled, y_train)

# Prediction
y_pred_linear = linear_svm.predict(X_test_scaled)

print("SVM WITH LINEAR KERNEL")
print("Accuracy:",accuracy_score(y_test, y_pred_linear))


cm_linear = confusion_matrix(
    y_test,
    y_pred_linear
)

print("Confusion Matrix:")
print(cm_linear)

ConfusionMatrixDisplay(
    confusion_matrix=cm_linear,
    display_labels=wine.target_names
).plot()

plt.title("Confusion Matrix - Linear SVM")
plt.show()

#----------------------------------------------
rbf_svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    random_state=42
)


rbf_svm.fit(X_train_scaled, y_train)
y_pred_rbf = rbf_svm.predict(X_test_scaled)

print("SVM WITH RBF KERNEL")


print("Accuracy:",
      accuracy_score(y_test, y_pred_rbf))


cm_rbf = confusion_matrix(
    y_test,
    y_pred_rbf
)

print("Confusion Matrix:")
print(cm_rbf)

ConfusionMatrixDisplay(
    confusion_matrix=cm_rbf,
    display_labels=wine.target_names
).plot()

plt.title("Confusion Matrix - RBF SVM")
plt.show()

#---------------------------------------------------

# SVM WITH POLYNOMIAL KERNEL

poly_svm = SVC(
    kernel="poly",
    degree=3,
    C=1.0,
    gamma="scale"
)

poly_svm.fit(X_train_scaled, y_train)
y_pred_poly = poly_svm.predict(X_test_scaled)

print("SVM WITH POLYNOMIAL KERNEL")

print("Accuracy:",
      accuracy_score(y_test, y_pred_poly))

cm_poly = confusion_matrix(
    y_test,
    y_pred_poly
)

print("Confusion Matrix:")
print(cm_poly)

ConfusionMatrixDisplay(
    confusion_matrix=cm_poly,
    display_labels=wine.target_names
).plot()

plt.title("Confusion Matrix - Polynomial SVM")
plt.show()



# MODEL COMPARISON

linear_accuracy = accuracy_score(
    y_test,
    y_pred_linear
)

rbf_accuracy = accuracy_score(
    y_test,
    y_pred_rbf
)

poly_accuracy = accuracy_score(
   y_test,
   y_pred_poly
)   


print("MODEL COMPARISON")
print("Linear SVM Accuracy:", linear_accuracy)
print("RBF SVM Accuracy:", rbf_accuracy)
print("Polynomial SVM Accuracy:", poly_accuracy)

if linear_accuracy > rbf_accuracy and linear_accuracy > poly_accuracy:
    print("\nLinear SVM performs better.")

elif rbf_accuracy > linear_accuracy and rbf_accuracy > poly_accuracy:
    print("\nRBF SVM performs better.")

elif poly_accuracy > linear_accuracy and poly_accuracy > rbf_accuracy:
    print("\nPolynomial SVM performs better.")

else:
    print("\nTwo or more models have the same highest accuracy.")
