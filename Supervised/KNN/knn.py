import numpy as np

# Euclidean distance between two images
def dist(img1, img2):
    return np.sqrt(np.sum((img1 - img2) ** 2))

# KNN classifier: predict the label of a single image
def classify_knn(image, k, train_images, train_labels):
    # Compute distances from the given image to all training images
    all_distances = [dist(image, train_image) for train_image in train_images]
    
    # Take indices of k nearest neighbors
    knn = np.argsort(all_distances)[:k]
    
    # Count the labels among k neighbors
    counts = np.bincount(train_labels[knn])
    
    # Return the most frequent label
    prediction = np.argmax(counts)
    return prediction

# KNN classifier with printing of neighbor labels (for debugging)
def classify_knn_print(image, k, train_images, train_labels):
    all_distances = [dist(image, train_image) for train_image in train_images]
    knn = np.argsort(all_distances)[:k]
    counts = np.bincount(train_labels[knn])
    print("Neighbor labels:", train_labels[knn])
    prediction = np.argmax(counts)
    return prediction


# Example usage (assuming train_images, train_labels, test_images, test_labels exist):
if __name__ == "__main__":
    # Predict single test samples
    print(f"Predicted class for the first image is {classify_knn(test_images[0], 5, train_images, train_labels)} "
          f"and the true label is {test_labels[0]}")
    
    print(f"Predicted class for the second image is {classify_knn(test_images[1], 5, train_images, train_labels)} "
          f"and the true label is {test_labels[1]}")

    # Predict all test samples
    test_predicted = np.array([classify_knn(img, 5, train_images, train_labels) for img in test_images])
    n_correct = np.sum(test_predicted == test_labels)
    knn_accuracy = (n_correct * 100) / len(test_labels)

    print(f"Final accuracy of our nearest neighbor classifier is {knn_accuracy:.2f}%")
