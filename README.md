# Project 3: Neural Network Comparison (Ames Housing)

This project is an extension of the initial "Ames Housing Price Prediction" assignment.
It aims to compare the performance of the original ensemble model against various
Neural Network architectures to determine the most effective approach for this dataset.

## 1. Project Goal
The objective was to evaluate whether Deep Learning techniques could outperform
traditional Machine Learning (Random Forest) on the Ames Housing regression task.
Key tasks included:
* **Optimizing the Baseline:** Hyperparameter tuning of the Random Forest model.
* **Neural Network Exploration:** Designing and testing multiple architectures (Simple, Deep, Regularized).
* **Performance Comparison:** Evaluating models based on R2 Score and RMSE.

---

## 2. Methodology

### A. Data Preprocessing
The data pipeline was updated to be compatible with Neural Networks:
* **Transformation:** Log-transformation of the target variable (`SalePrice`) to normalize distribution.
* **Scaling:** `StandardScaler` applied to all numerical features (critical for Neural Network convergence).
* **Encoding:** `OneHotEncoder` applied to all categorical features.
* **Input Shape:** Final processed feature set contained 269 inputs.

### B. Models Evaluated

**1. Baseline: Optimized Random Forest**
* **Improvement:** Used `RandomizedSearchCV` to tune `n_estimators`, `max_depth`, and `min_samples_leaf`.
* **Result:** The robust default parameters were found to be statistically optimal, confirming the model's stability.

**2. Neural Networks (TensorFlow/Keras)**
We experimented with four architectures to address overfitting and instability:
* **Model 1 (Simple):** A shallow network with 1 hidden layer (64 neurons).
* **Model 2 (Deep):** A deep network with 3 hidden layers (128 -> 64 -> 32 neurons).
* **Model 3 (Regularized):** Added Dropout layers and Early Stopping to the deep network.
* **Model 4 (Robust):** Added **Batch Normalization** to stabilize gradients and improve convergence.

---

## 3. Results & Comparison

The comparison revealed that for this specific tabular dataset, the **Random Forest** significantly outperformed all Neural Network variants.

| Model | Test R2 Score | Test RMSE ($) |
|:---|---:|---:|
| **Optimized Random Forest** | **0.9000** | **$30,646** |
| Robust NN (Batch Norm) | 0.7239 | $55,457 |
| Simple NN (1 Layer) | 0.6698 | $127,143 |
| Deep NN (3 Layers) | 0.2383 | $945,035 |
| Regularized NN | -3.5741 | (Failed) |

### Key Findings
1.  **Tabular Data Dominance:** Random Forest proved superior for this structured dataset with limited samples (~1400 rows). Neural Networks struggled to generalize without massive data.
2.  **Complexity Trap:** Increasing network depth (Model 2 & 3) actually *hurt* performance due to instability and overfitting.
3.  **Stabilization:** Adding Batch Normalization (Model 4) successfully rescued the Neural Network, boosting R2 from negative values to 0.72, though it still lagged behind the ensemble approach.

---

## 4. How to Run
1.  Open `ML_Project_2_Neural_Networks.ipynb` in Google Colab.
2.  Upload `train.csv` and `data_description.txt`.
3.  Run all cells to replicate the training of the Baseline RF and all 4 Neural Networks.
