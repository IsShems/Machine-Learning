import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Euclidean distance
def dist(img1, img2):
    return np.sqrt(np.sum((img1 - img2) ** 2))

# KNN classifier
def classify_knn(image, k, train_images, train_labels):
    all_distances = [dist(image, train_image) for train_image in train_images]
    knn = np.argsort(all_distances)[:k]
    counts = np.bincount(train_labels[knn])
    return np.argmax(counts)

# Cross-validation
def cross_validate_knn(k, images, labels, num_folds=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    accuracies = []

    for train_index, val_index in kf.split(images):
        X_train, X_val = images[train_index], images[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        y_pred = [classify_knn(img, k, X_train, y_train) for img in X_val]
        acc = accuracy_score(y_val, y_pred)
        accuracies.append(acc)

    return np.mean(accuracies)

# Example usage (assuming images, labels exist):
if __name__ == "__main__":
    mean_accuracy = cross_validate_knn(5, train_images, train_labels, num_folds=5)
    print(f"Average accuracy with 5-fold cross-validation: {mean_accuracy:.2f}")
