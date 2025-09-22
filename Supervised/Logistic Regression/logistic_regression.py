# Logistic Regression Classifier
# ------------------------------
# In this notebook, we implement and evaluate Logistic Regression
# on the MNIST dataset using scikit-learn.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

# Load dataset (MNIST digits)
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

# Convert labels to integers
y = y.astype(np.uint8)

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize Logistic Regression
# 'lbfgs' solver works well for multiclass problems
clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial")

# Train the model
clf.fit(X_train, y_train)

# Evaluate with cross-validation
cv_scores = cross_val_score(clf, X_train, y_train, cv=3)
print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", np.mean(cv_scores))

# Test set evaluation
test_accuracy = clf.score(X_test, y_test)
print("Test accuracy:", test_accuracy)
