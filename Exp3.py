import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score


# ============================================================
# Step 2: Load Iris Dataset
# ============================================================

iris = load_iris()

X = iris.data
y = iris.target

print("Dataset Shape:", X.shape)

print("\nFeature Names:")
print(iris.feature_names)

print("\nTarget Names:")
print(iris.target_names)


# ============================================================
# Step 3: Split Dataset
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# Step 4: Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# Step 5: Create Logistic Regression Model
# ============================================================

model = LogisticRegression(max_iter=1000)


# ============================================================
# Step 6: Define Hyperparameters
# ============================================================
parameters = [
    {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs'],
        'penalty': ['l2'],
        'max_iter': [100, 500, 1000, 2000]
    },
    {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear'],
        'penalty': ['l1', 'l2'],
        'max_iter': [100, 500, 1000, 2000]
    }
]

# ============================================================
# Step 7: Grid Search
# ============================================================

grid_search = GridSearchCV(
    model,
    parameters,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)


print("\n========== GRID SEARCH ==========")

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(grid_search.best_score_)


# Test the best model

grid_prediction = grid_search.predict(X_test)

grid_accuracy = accuracy_score(
    y_test,
    grid_prediction
)

print("\nGrid Search Test Accuracy:")
print(grid_accuracy)


# ============================================================
# Step 8: Random Search
# ============================================================

random_search = RandomizedSearchCV(
    model,
    parameters,
    n_iter=5,
    cv=5,
    scoring='accuracy',
    random_state=42
)

random_search.fit(X_train, y_train)


print("\n========== RANDOM SEARCH ==========")

print("Best Parameters:")
print(random_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(random_search.best_score_)


# Test the best model

random_prediction = random_search.predict(X_test)

random_accuracy = accuracy_score(
    y_test,
    random_prediction
)

print("\nRandom Search Test Accuracy:")
print(random_accuracy)


# ============================================================
# Step 9: Compare Results
# ============================================================

print("\n========== COMPARISON ==========")

print("Grid Search Accuracy  :", grid_accuracy)
print("Random Search Accuracy:", random_accuracy)