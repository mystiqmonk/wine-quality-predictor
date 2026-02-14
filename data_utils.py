"""
Data loading and preprocessing utilities for Wine Quality classification.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

def load_wine_data():
    """Load and preprocess Wine Quality dataset from UCI."""
    # Load red and white wine
    red = pd.read_csv(DATA_DIR / "winequality-red.csv", sep=";")
    white = pd.read_csv(DATA_DIR / "winequality-white.csv", sep=";")
    
    # Add wine type feature (0=red, 1=white) - gives us 12 features total
    red["wine_type"] = 0
    white["wine_type"] = 1
    
    # Combine datasets
    df = pd.concat([red, white], ignore_index=True)
    
    # Binary classification: quality >= 6 is "good" (1), else "bad" (0)
    df["quality_binary"] = (df["quality"] >= 6).astype(int)
    
    X = df.drop(columns=["quality", "quality_binary"])
    y = df["quality_binary"]
    
    return X, y

def get_train_test_split(test_size=0.2, random_state=42):
    """Get train-test split."""
    from sklearn.model_selection import train_test_split
    
    X, y = load_wine_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
