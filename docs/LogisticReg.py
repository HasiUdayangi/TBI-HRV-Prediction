import os
import pickle
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from utils.performance_analysis import bootstrap_metrics, plot_confusion_matrix, call_precision_recall_curve


def load_data(data_dir=""):
    """Loads pre-saved training and test data from pickle files."""
    with open(os.path.join(data_dir, "X_train_val.pkl"), "rb") as f:
        X_train_val = pickle.load(f)
    with open(os.path.join(data_dir, "X_test.pkl"), "rb") as f:
        X_test = pickle.load(f)
    with open(os.path.join(data_dir, "y_train_val.pkl"), "rb") as f:
        y_train_val = pickle.load(f)
    with open(os.path.join(data_dir, "y_test.pkl"), "rb") as f:
        y_test = pickle.load(f)

    return X_train_val, X_test, y_train_val, y_test

def preprocess_data(X_train_val, X_test):
    X_train_reshaped = X_train_val.reshape(X_train_val.shape[0], -1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

    return X_train_reshaped, X_test_reshaped

def train_logistic_regression(X_train_reshaped, y_train_val, n_splits=5):
    """Trains a logistic regression model using stratified K-Fold cross-validation."""
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_val), y=y_train_val)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    auc_scores = []
    models = []

    for train_idx, val_idx in kf.split(X_train_reshaped, y_train_val):
        X_train, X_val = X_train_reshaped[train_idx], X_train_reshaped[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

        # Apply SMOTE for handling class imbalance
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Train Logistic Regression Model
        model = LogisticRegression(class_weight=class_weight_dict, max_iter=500, solver="liblinear")
        model.fit(X_train, y_train)

        # Predict probabilities
        y_val_pred_prob = model.predict_proba(X_val)[:, 1]

        # Evaluate Performance Metrics
        val_auc = roc_auc_score(y_val, y_val_pred_prob)
        auc_scores.append(val_auc)
        models.append(model)

        print(f'Fold AUC: {val_auc:.4f}')

    # Select best model
    best_model = models[np.argmax(auc_scores)]
    joblib.dump(best_model, "models/logistic_regression.pkl")

    return best_model, np.mean(auc_scores)


def evaluate_model(model, X_test_reshaped, y_test):
    """Evaluates the trained model on the test dataset."""
    y_pred_prob_test = model.predict_proba(X_test_reshaped)[:, 1]
    y_pred_test = (y_pred_prob_test > 0.5).astype(int)

    auc_score = roc_auc_score(y_test, y_pred_prob_test)
    accuracy = accuracy_score(y_test, y_pred_test)

    print("\n🔹 Logistic Regression Model Performance on Test Set:")
    print(f"AUC: {auc_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred_test))

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plot_confusion_matrix(cm, ['HIGH RISK', 'LOW RISK'])

    # Bootstrap Performance Analysis
    metric_results = bootstrap_metrics(y_test, y_pred_prob_test)
    for metric, values in metric_results.items():
        print(f"{metric}: Mean = {values['Mean']:.4f}, 95% CI = {values['95% CI'][0]:.4f} - {values['95% CI'][1]:.4f}")

    # Precision-Recall Curve
    call_precision_recall_curve(y_test, y_pred_test, y_pred_prob_test, 1)

    # Save Results
    results_df = pd.DataFrame({
        "AUC": [auc_score],
        "Accuracy": [accuracy],
    })
    results_df.to_csv("results/logistic_regression_results.csv", index=False)

    return auc_score, accuracy


def main():
    """Runs the full pipeline for training and evaluating logistic regression."""
    print("\n🚀 Loading Data...")
    X_train_val, X_test, y_train_val, y_test = load_data()

    print("\n🔹 Preprocessing Data...")
    X_train_reshaped, X_test_reshaped = preprocess_data(X_train_val, X_test)

    print("\n🔹 Training Logistic Regression with Cross-Validation...")
    best_model, avg_auc = train_logistic_regression(X_train_reshaped, y_train_val)

    print(f"\n✅ Best Cross-Validation AUC: {avg_auc:.4f}")

    print("\n🔹 Evaluating Model on Test Data...")
    evaluate_model(best_model, X_test_reshaped, y_test)


if __name__ == "__main__":
    main()
