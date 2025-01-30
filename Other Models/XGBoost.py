import numpy as np
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, confusion_matrix
from performance_analysis import plot_confusion_matrix, bootstrap_metrics, plot_roc_curve

SEED = 42
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'colsample_bytree': [0.3, 0.7, 1.0]
}

def train_xgboost(X_train, y_train):
    """
    Perform hyperparameter tuning and train the best XGBoost model.
    """
    xgb = XGBClassifier(use_label_encoder=False, random_state=SEED)
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation AUC: {:.4f}".format(grid_search.best_score_))
    
    return grid_search.best_estimator_

def evaluate_xgboost(X_train, y_train, X_test, y_test):
    """
    Train and evaluate XGBoost using Stratified K-Fold validation.
    """
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_params = {
        'n_estimators': 100, 
        'learning_rate': 0.01, 
        'max_depth': 3,  
        'colsample_bytree': 0.7
    }

    auc_scores = []
    models = []

    for train_index, test_index in kf.split(X_train, y_train):
        X_tr, X_val = X_train[train_index], X_train[test_index]
        y_tr, y_val = y_train[train_index], y_train[test_index]

        model = XGBClassifier(**best_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=True)
        pred_val = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, pred_val)

        auc_scores.append(auc)
        models.append(model)
        print(f'Fold AUC: {auc}')

    best_model = models[np.argmax(auc_scores)]
    print(f'Average AUC: {np.mean(auc_scores)}')

    # Save best model
    joblib.dump(best_model, 'best_xgboost_model.pkl')

    # Test Set Evaluation
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    test_auc = roc_auc_score(y_test, y_pred_prob)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Set AUC: {test_auc:.4f}")
    print(f"Test Set Accuracy: {test_accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Generate Performance Metrics
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, ['HIGH RISK', 'LOW RISK'])
    plot_roc_curve(y_test, y_pred_prob)
    
    metric_results = bootstrap_metrics(y_test, y_pred)
    for metric, values in metric_results.items():
        print(f"{metric}: Mean = {values['Mean']:.4f}, 95% CI = {values['95% CI'][0]:.4f} - {values['95% CI'][1]:.4f}")

    return best_model


