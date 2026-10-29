import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
# We use the classic Iris dataset (multiclass target: 0, 1, or 2)
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# DECISION TREE CLASSIFIER
# ==========================================
print("--- Decision Tree Classifier ---")

# Initialize and train the model
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Make predictions
dt_predictions = dt_classifier.predict(X_test)

# Evaluate performance
dt_accuracy = accuracy_score(y_test, dt_predictions)
print(f"Accuracy: {dt_accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, dt_predictions, target_names=iris.target_names))


# ==========================================
# RANDOM FOREST CLASSIFIER
# ==========================================
print("\n--- Random Forest Classifier ---")

# Initialize and train the model (using 100 decision trees)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_classifier.predict(X_test)

# Evaluate performance
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Accuracy: {rf_accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, rf_predictions, target_names=iris.target_names))


# ==========================================
# EXTRA: Visualize the Decision Tree
# ==========================================
plt.figure(figsize=(12, 8))
plot_tree(dt_classifier, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Visualizing the Trained Decision Tree")
plt.show()