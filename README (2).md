# K-Nearest Neighbor (KNN) on MNIST Data

## Project Overview

This project implements a **K-Nearest Neighbor (KNN)** classifier to recognize handwritten digits from the MNIST dataset. It demonstrates the end-to-end workflow of a machine learning project: data loading, train-validation-test splitting, hyperparameter tuning, model evaluation, and visual prediction inspection.

## Dataset

- **Source**: scikit-learn's digits dataset (smaller subset of the full MNIST)
- **Total samples**: 1,797 digit images
- **Image size**: 8×8 grayscale (flattened to 64-dimensional vectors)
- **Classes**: Digits 0–9 (10 classes)
- **Data split**:
  - Training: 67.5% (1,212 samples)
  - Validation: 7.5% (135 samples)
  - Testing: 25% (450 samples)

## Project Structure

### 1. **Data Loading & Splitting**
```python
# Load MNIST digits
mnist = datasets.load_digits()

# First split: 75% train, 25% test
(trainData, testData, trainLabels, testLabels) = train_test_split(...)

# Second split: Extract 10% of training as validation
(trainData, valData, trainLabels, valLabels) = train_test_split(...)
```

### 2. **Hyperparameter Tuning**
The notebook tests **k values from 1 to 29** (odd numbers only) on the validation set to find the optimal k.

**Results**:
- **Best k**: 1 (ties with k=3 to k=15, but k=1 is most efficient)
- **Validation accuracy**: 99.26%

### 3. **Model Training & Testing**
The optimal k is used to train the final model on the combined training data, then evaluated on the test set.

**Test Set Results**:
- **Overall accuracy**: 98%
- **Perfect digits** (100% accuracy): 0, 2, 6, 7
- **Lowest accuracy**: Digit 1 (95%)

### 4. **Visualization**
The notebook displays 6 random predictions from the test set, highlighting correct predictions (green) and misclassifications (red).

---

## Key Corrections Made

### Issue 1: Incomplete Visualization Loop
**Original Problem**: The last cell had a broken loop structure split across multiple cells, making variable references undefined.

**Fix**: Consolidated the prediction visualization into a single, complete loop using matplotlib subplots that:
- Selects 6 random test samples
- Makes predictions for each
- Displays the 8×8 images in a 2×3 grid
- Shows prediction vs. actual label with color coding

### Issue 2: Range Object Indexing
**Original Problem**: `kVals` was a `range` object, which can't be indexed with `kVals[i]`

**Fix**: Convert to list before indexing:
```python
kVals_list = list(kVals)
model = KNeighborsClassifier(n_neighbors=kVals_list[i])
```

### Issue 3: Missing Imports
**Original**: TensorFlow was imported but never used (unnecessary dependency)

**Fix**: Removed unused import to simplify dependencies.

### Issue 4: Incomplete Code Cells
**Original**: Several cells had trailing whitespace and incomplete logic

**Fix**: Cleaned up all cells and ensured complete, executable code.

---

## Dependencies

```
numpy
scikit-learn
scikit-image
matplotlib
opencv-python (cv2)
imutils
```

### Installation
```bash
pip install numpy scikit-learn scikit-image matplotlib opencv-python imutils
```

---

## How to Run

1. **Open the notebook**:
   ```bash
   jupyter notebook KNN_on_MNIST_data_CORRECTED.ipynb
   ```

2. **Execute cells in order** (Kernel → Run All Cells)

3. **Expected output**:
   - Data split statistics
   - k-value accuracy comparisons
   - Classification report (precision, recall, F1-score)
   - Grid of 6 sample predictions with actual vs. predicted labels

---

## Algorithm: K-Nearest Neighbor

### How it Works
1. **Store** all training data (lazy learning)
2. **For a new sample**, compute distance to all training points
3. **Find k nearest neighbors**
4. **Predict** the most common class among those k neighbors

### Distance Metric
- Default: **Euclidean distance** (L2 norm)
- Formula: `d = √(Σ(x_i - y_i)²)`

### Advantages
- ✅ Simple and interpretable
- ✅ No training phase (fast to fit)
- ✅ Naturally handles multi-class problems
- ✅ Effective on well-preprocessed data like MNIST

### Disadvantages
- ❌ Slow prediction (must compute distance to all training points)
- ❌ High memory usage
- ❌ Sensitive to irrelevant features
- ❌ Requires careful feature scaling

---

## Results & Analysis

### Validation Accuracy by k
| k  | Accuracy |
|----|----------|
| 1  | 99.26%   |
| 3  | 99.26%   |
| 5  | 99.26%   |
| 7  | 99.26%   |
| 9  | 99.26%   |
| ... | ...      |
| 29 | 97.04%   |

**Key insight**: k=1 performs best because the data is well-separated; larger k values introduce noise from distant neighbors.

### Test Set Performance by Digit
| Digit | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 1.00      | 1.00   | 1.00     | 43      |
| 1     | 0.95      | 1.00   | 0.97     | 37      |
| 2     | 1.00      | 1.00   | 1.00     | 38      |
| 3     | 0.98      | 0.98   | 0.98     | 46      |
| 4     | 0.98      | 0.98   | 0.98     | 55      |
| 5     | 0.98      | 1.00   | 0.99     | 59      |
| 6     | 1.00      | 1.00   | 1.00     | 45      |
| 7     | 1.00      | 0.98   | 0.99     | 41      |
| 8     | 0.97      | 0.95   | 0.96     | 38      |
| 9     | 0.96      | 0.94   | 0.95     | 48      |

---

## Limitations & Real-World Considerations

1. **MNIST is "easy"**: The dataset is heavily preprocessed (centered, normalized, 28×28 grayscale). Real-world handwriting is messier.

2. **No feature engineering**: Uses raw pixel intensities. In production, preprocessing and feature extraction (edge detection, strokes) would improve robustness.

3. **Slow inference**: KNN is impractical for large-scale deployment. Deep learning (CNNs) or other fast classifiers are preferred.

4. **Curse of dimensionality**: With 64 features, the distance metric becomes less meaningful. High-dimensional spaces are sparsely populated.

---

## Extending the Project

### 1. Improve Accuracy
- Try different distance metrics (Manhattan, Cosine, Mahalanobis)
- Use dimensionality reduction (PCA, t-SNE)
- Apply feature normalization

### 2. Speed Up Prediction
- Use KD-trees or Ball trees for faster neighbor search
- Implement caching or approximate nearest neighbors (ANN)

### 3. Compare Models
- Logistic Regression
- Random Forest
- Support Vector Machines (SVM)
- Convolutional Neural Networks (CNN)

### 4. Use Full MNIST Dataset
- Download from http://yann.lecun.com/exdb/mnist/
- 60,000 training + 10,000 test samples
- More challenging; will expose KNN's limitations

---

## References

- **scikit-learn KNN**: https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors
- **MNIST Dataset**: http://yann.lecun.com/exdb/mnist/
- **KNN Algorithm**: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

---

## Author Notes

This notebook is an educational demonstration of KNN applied to a canonical dataset. While it achieves 98% accuracy, this reflects the ideal nature of the MNIST data. In production, expect:
- Lower accuracy on real-world handwritten digits
- Need for preprocessing and augmentation
- Trade-offs between accuracy, speed, and memory

KNN is a valuable baseline and educational tool, but for deployment-ready systems, modern deep learning approaches are strongly recommended.

---

## License

This project is provided as educational material. The MNIST dataset is in the public domain.
