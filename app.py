"""
Wine Quality Classification - Interactive Streamlit App
Demonstrates 6 ML models for binary wine quality prediction (Good vs Bad)
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Page config
st.set_page_config(page_title="Wine Quality Classifier", page_icon="🍷", layout="wide")

# Feature names (must match training data order)
FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type"
]

@st.cache_resource
def load_models():
    """Load all saved models and scalers."""
    model_dir = Path(__file__).parent / "model"
    models = {}
    
    # Logistic Regression (uses scaler)
    try:
        models["logistic_regression"] = {
            "model": joblib.load(model_dir / "logistic_regression.pkl"),
            "scaler": joblib.load(model_dir / "scaler.pkl"),
            "needs_scaling": True
        }
    except FileNotFoundError:
        pass
    
    # Decision Tree (no scaling)
    try:
        models["decision_tree"] = {
            "model": joblib.load(model_dir / "decision_tree.pkl"),
            "scaler": None,
            "needs_scaling": False
        }
    except FileNotFoundError:
        pass
    
    # KNN (uses scaler_knn)
    try:
        models["knn"] = {
            "model": joblib.load(model_dir / "knn.pkl"),
            "scaler": joblib.load(model_dir / "scaler_knn.pkl"),
            "needs_scaling": True
        }
    except FileNotFoundError:
        pass
    
    # Naive Bayes (uses scaler_nb)
    try:
        models["naive_bayes"] = {
            "model": joblib.load(model_dir / "naive_bayes.pkl"),
            "scaler": joblib.load(model_dir / "scaler_nb.pkl"),
            "needs_scaling": True
        }
    except FileNotFoundError:
        pass
    
    # Random Forest (no scaling)
    try:
        models["random_forest"] = {
            "model": joblib.load(model_dir / "random_forest.pkl"),
            "scaler": None,
            "needs_scaling": False
        }
    except FileNotFoundError:
        pass
    
    # XGBoost (no scaling)
    try:
        models["xgboost"] = {
            "model": joblib.load(model_dir / "xgboost.pkl"),
            "scaler": None,
            "needs_scaling": False
        }
    except FileNotFoundError:
        pass
    
    return models

def predict(model_info, X):
    """Run prediction with optional scaling."""
    model = model_info["model"]
    if model_info["needs_scaling"]:
        X = model_info["scaler"].transform(X)
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        proba = pred
    return pred, proba

def main():
    st.title("🍷 Wine Quality Classification App")
    st.markdown("""
    Predict wine quality (Good/Bad) using physicochemical properties. 
    **Good** = quality score ≥ 6, **Bad** = quality score < 6.
    """)
    
    models = load_models()
    if not models:
        st.error("No models found! Run `python train_all_models.py` first to train and save models.")
        return
    
    # Sidebar - Model selection & input
    st.sidebar.header("Configuration")
    model_choice = st.sidebar.selectbox(
        "Select Model",
        list(models.keys()),
        format_func=lambda x: x.replace("_", " ").title()
    )
    
    st.sidebar.subheader("Input Features")
    inputs = {}
    
    # Sliders for numerical features
    col1, col2 = st.sidebar.columns(2)
    with col1:
        inputs["fixed acidity"] = st.slider("Fixed acidity", 4.0, 16.0, 7.0, 0.1)
        inputs["volatile acidity"] = st.slider("Volatile acidity", 0.0, 1.6, 0.3, 0.01)
        inputs["citric acid"] = st.slider("Citric acid", 0.0, 1.0, 0.3, 0.01)
        inputs["residual sugar"] = st.slider("Residual sugar", 0.5, 16.0, 2.5, 0.1)
        inputs["chlorides"] = st.slider("Chlorides", 0.01, 0.6, 0.08, 0.01)
        inputs["free sulfur dioxide"] = st.slider("Free sulfur dioxide", 1, 72, 15, 1)
    with col2:
        inputs["total sulfur dioxide"] = st.slider("Total sulfur dioxide", 6, 440, 45, 1)
        inputs["density"] = st.slider("Density", 0.99, 1.04, 0.996, 0.001)
        inputs["pH"] = st.slider("pH", 2.7, 4.0, 3.2, 0.01)
        inputs["sulphates"] = st.slider("Sulphates", 0.2, 2.0, 0.5, 0.01)
        inputs["alcohol"] = st.slider("Alcohol (%)", 8.0, 15.0, 10.5, 0.1)
        inputs["wine_type"] = st.radio("Wine type", [0, 1], format_func=lambda x: "Red" if x == 0 else "White", index=0)
    
    # Build feature vector in correct order
    X = np.array([[inputs[f] for f in FEATURES]])
    
    # Prediction
    model_info = models[model_choice]
    pred, proba = predict(model_info, X)
    
    # Results
    st.subheader("Prediction Result")
    result = "Good 🟢" if pred[0] == 1 else "Bad 🔴"
    st.markdown(f"### {result}")
    st.progress(min(float(proba[0]), 1.0))
    st.caption(f"Predicted probability of Good wine: {proba[0]:.2%}")
    
    # Tabs: Compare models, Dataset info, Evaluation metrics
    tab1, tab2, tab3 = st.tabs(["Compare All Models", "Dataset Info", "Evaluation Metrics"])
    
    with tab1:
        st.subheader("Compare Predictions Across Models")
        results = []
        for name, info in models.items():
            p, pr = predict(info, X)
            results.append({
                "Model": name.replace("_", " ").title(),
                "Prediction": "Good" if p[0] == 1 else "Bad",
                "Probability": f"{pr[0]:.2%}"
            })
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    
    with tab2:
        st.markdown("""
        ### Wine Quality Dataset (UCI)
        - **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)
        - **Instances:** 6,497 (red + white wine)
        - **Features:** 12 (11 physicochemical + wine type)
        - **Task:** Binary classification (Quality ≥ 6 → Good, else Bad)
        
        **Features:** fixed acidity, volatile acidity, citric acid, residual sugar,
        chlorides, free sulfur dioxide, total sulfur dioxide, density, pH,
        sulphates, alcohol, wine_type (0=red, 1=white)
        """)
    
    with tab3:
        st.subheader("Model Performance (Test Set)")
        metrics_df = pd.DataFrame({
            "ML Model": ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"],
            "Accuracy": [0.7392, 0.7654, 0.7408, 0.6815, 0.8485, 0.8285],
            "AUC": [0.8057, 0.7486, 0.8004, 0.7419, 0.9055, 0.8825],
            "Precision": [0.7665, 0.8166, 0.7780, 0.7216, 0.8598, 0.8513],
            "Recall": [0.8457, 0.8117, 0.8262, 0.8092, 0.9089, 0.8834],
            "F1": [0.8042, 0.8141, 0.8014, 0.7629, 0.8836, 0.8670],
            "MCC": [0.4214, 0.4961, 0.4308, 0.2873, 0.6690, 0.6265]
        })
        st.dataframe(metrics_df.style.highlight_max(axis=0, subset=metrics_df.columns[1:]), use_container_width=True)

if __name__ == "__main__":
    main()
