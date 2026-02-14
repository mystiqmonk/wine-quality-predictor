"""Train all models and save them. Run this to generate model files for app.py"""
import sys
from pathlib import Path
from data_utils import get_train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

Path("model").mkdir(exist_ok=True)
X_train, X_test, y_train, y_test = get_train_test_split()

def eval_print(name, y_pred, y_pred_proba=None):
    if y_pred_proba is None:
        y_pred_proba = y_pred  # For models without proba
    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("AUC:", round(roc_auc_score(y_test, y_pred_proba), 4))
    print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
    print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 4))
    print("F1:", round(f1_score(y_test, y_pred, zero_division=0), 4))
    print("MCC:", round(matthews_corrcoef(y_test, y_pred), 4))

# 1. Logistic Regression
scaler_lr = StandardScaler()
X_tr_s = scaler_lr.fit_transform(X_train)
X_te_s = scaler_lr.transform(X_test)
m1 = LogisticRegression(max_iter=1000, random_state=42)
m1.fit(X_tr_s, y_train)
joblib.dump(m1, "model/logistic_regression.pkl")
joblib.dump(scaler_lr, "model/scaler.pkl")
eval_print("Logistic Regression", m1.predict(X_te_s), m1.predict_proba(X_te_s)[:, 1])

# 2. Decision Tree
m2 = DecisionTreeClassifier(random_state=42)
m2.fit(X_train, y_train)
joblib.dump(m2, "model/decision_tree.pkl")
eval_print("Decision Tree", m2.predict(X_test), m2.predict_proba(X_test)[:, 1])

# 3. KNN
scaler_knn = StandardScaler()
X_tr_s = scaler_knn.fit_transform(X_train)
X_te_s = scaler_knn.transform(X_test)
m3 = KNeighborsClassifier(n_neighbors=5)
m3.fit(X_tr_s, y_train)
joblib.dump(m3, "model/knn.pkl")
joblib.dump(scaler_knn, "model/scaler_knn.pkl")
eval_print("KNN", m3.predict(X_te_s), m3.predict_proba(X_te_s)[:, 1])

# 4. Naive Bayes
scaler_nb = StandardScaler()
X_tr_s = scaler_nb.fit_transform(X_train)
X_te_s = scaler_nb.transform(X_test)
m4 = GaussianNB()
m4.fit(X_tr_s, y_train)
joblib.dump(m4, "model/naive_bayes.pkl")
joblib.dump(scaler_nb, "model/scaler_nb.pkl")
eval_print("Naive Bayes", m4.predict(X_te_s), m4.predict_proba(X_te_s)[:, 1])

# 5. Random Forest
m5 = RandomForestClassifier(n_estimators=100, random_state=42)
m5.fit(X_train, y_train)
joblib.dump(m5, "model/random_forest.pkl")
eval_print("Random Forest", m5.predict(X_test), m5.predict_proba(X_test)[:, 1])

# 6. XGBoost
if HAS_XGB:
    m6 = XGBClassifier(n_estimators=100, random_state=42)
    m6.fit(X_train, y_train)
    joblib.dump(m6, "model/xgboost.pkl")
    eval_print("XGBoost", m6.predict(X_test), m6.predict_proba(X_test)[:, 1])
else:
    print("\nXGBoost skipped - install with: pip install xgboost")

print("\nAll models saved to model/")
