# K-Nearest Neighbors (KNN)

This folder contains an implementation of the **K-Nearest Neighbors (KNN)** algorithm written **from scratch** (without using `scikit-learn`).  
The classifier works on image data and calculates the Euclidean distance between samples.

## Files
- `knn.py` — Python script with the full KNN implementation.
- `knn.ipynb` — Jupyter Notebook version (same code, with cells for step-by-step execution).

## Code structure
1. **Distance function**  
   - `dist(img1, img2)` → computes the Euclidean distance between two images.
2. **KNN classifier**  
   - `classify_knn(image, k, train_images, train_labels)` → predicts the label of a single image.  
   - `classify_knn_print(...)` → same as above, but prints the neighbor labels (useful for debugging).
3. **Evaluation**  
   - The script shows predictions for the first few test samples.
   - Then it predicts all test samples and calculates the final accuracy.

## How to run
### Option 1: Python script
Make sure you have `numpy` installed:
```bash
pip install numpy

