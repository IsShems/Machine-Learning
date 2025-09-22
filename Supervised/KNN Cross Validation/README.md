# K-Nearest Neighbors with Cross-Validation

This notebook demonstrates **K-Nearest Neighbors (KNN)** combined with **k-fold cross-validation** for model evaluation.

## Files
- `knn_cross_validation.py` — Python script implementing KNN with cross-validation.
- `knn_cross_validation.ipynb` — Jupyter Notebook version (step-by-step execution).

## Code structure
1. **Distance function**  
   - `dist(img1, img2)` → Euclidean distance.
2. **KNN classifier**  
   - `classify_knn(image, k, train_images, train_labels)` → predicts label for one sample.
3. **Cross-validation**  
   - `cross_validate_knn(k, images, labels, num_folds)` → evaluates KNN accuracy using k-fold CV.

## How to run
### Python script
```bash
pip install numpy scikit-learn
python knn_cross_validation.py
