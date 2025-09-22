# Decision Tree Classifier
# ------------------------
# In this notebook, we implement and evaluate a Decision Tree classifier 
# on the MNIST dataset. We use scikit-learn for simplicity.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Load dataset (MNIST digits)
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

# Convert labels to integers
y = y.astype(np.uint8)

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Evaluate with cross-validation
cv_scores = cross_val_score(clf, X_train, y_train, cv=3)
print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", np.mean(cv_scores))

# Test set evaluation
test_accuracy = clf.score(X_test, y_test)
print("Test accuracy:", test_accuracy)
