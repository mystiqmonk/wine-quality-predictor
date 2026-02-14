# Wine Quality Classification - ML Assignment 2

Interactive Streamlit web application demonstrating 6 machine learning classification models for predicting wine quality (Good vs Bad) from physicochemical properties.

**[🚀 Live App on Streamlit Community Cloud](https://wine-quality-predictor-2025aa05049.streamlit.app)** 

---

## a. Problem Statement

Predict whether a wine is of **good** or **bad** quality based on 12 physicochemical measurements. This is a **binary classification** problem where:
- **Good** = quality score ≥ 6  
- **Bad** = quality score < 6  

The goal is to compare the performance of six different ML classifiers on this task and deploy an interactive demo for evaluation.

---

## b. Dataset Description

- **Name:** Wine Quality Dataset (UCI)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)
- **Type:** Portuguese "Vinho Verde" wine samples (red and white variants)
- **Total instances:** 6,497 (1,599 red + 4,898 white)
- **Features:** 12
  1. Fixed acidity
  2. Volatile acidity
  3. Citric acid
  4. Residual sugar
  5. Chlorides
  6. Free sulfur dioxide
  7. Total sulfur dioxide
  8. Density
  9. pH
  10. Sulphates
  11. Alcohol
  12. Wine type (0 = red, 1 = white)
- **Target:** Binary (quality ≥ 6 → 1, else 0)
- **Preprocessing:** Combined red and white wine; added wine type as feature; stratified train-test split (80-20)

---

## c. Models Used & Evaluation Metrics

### Comparison Table

| ML Model Name      | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|--------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression| 0.7392   | 0.8057| 0.7665    | 0.8457 | 0.8042| 0.4214|
| Decision Tree      | 0.7654   | 0.7486| 0.8166    | 0.8117 | 0.8141| 0.4961|
| kNN                | 0.7408   | 0.8004| 0.7780    | 0.8262 | 0.8014| 0.4308|
| Naive Bayes        | 0.6815   | 0.7419| 0.7216    | 0.8092 | 0.7629| 0.2873|
| Random Forest      | 0.8485   | 0.9055| 0.8598    | 0.9089 | 0.8836| 0.6690|
| XGBoost            | 0.8285   | 0.8825| 0.8513    | 0.8834 | 0.8670| 0.6265|

### Model Performance Observations

| ML Model Name      | Observation about model performance |
|--------------------|--------------------------------------|
| Logistic Regression| Provides a solid baseline with good AUC (0.81) and recall. Simple, interpretable, and benefits from feature scaling. |
| Decision Tree      | Slightly better accuracy than Logistic Regression without scaling. Moderate AUC; prone to overfitting on complex boundaries. |
| kNN                | Similar to Logistic Regression; sensitive to feature scaling. Performance depends on choice of k and distance metric. |
| Naive Bayes        | Lowest overall performance. Assumption of feature independence may not hold for correlated physicochemical variables. |
| Random Forest      | **Best performer** across most metrics (highest Accuracy, AUC, F1, MCC). Ensemble reduces overfitting and captures non-linear patterns well. |
| XGBoost            | Second-best; strong AUC and F1. Gradient boosting is effective; slightly behind Random Forest on this dataset. |

---

## Repository Structure

```
project-folder/
├── app.py                    # Streamlit web application
├── requirements.txt
├── README.md
├── data_utils.py             # Data loading & preprocessing
├── train_all_models.py       # Script to train & save all models
├── data/
│   ├── winequality-red.csv
│   └── winequality-white.csv
├── model/                    # Saved model files (*.pkl)
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   ├── scaler_knn.pkl
│   └── scaler_nb.pkl
├── 1_logistic_regression.ipynb
├── 2_decision_tree.ipynb
├── 3_knn.ipynb
├── 4_naive_bayes.ipynb
├── 5_random_forest.ipynb
└── 6_xgboost.ipynb
```

---

## Setup & Run

### Local

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train models (if not already done):
   ```bash
   python train_all_models.py
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open http://localhost:8501 in your browser.

### Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub repo and select `app.py` as the main file.
4. Ensure `requirements.txt` is in the repo root.
5. Deploy and share the generated URL.

---

## Clickable Links

- **Streamlit App:** https://wine-quality-predictor-2025aa05049.streamlit.app
- **Dataset (UCI):** [Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)
