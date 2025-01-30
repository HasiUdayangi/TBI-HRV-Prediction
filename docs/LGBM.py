import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report, confusion_matrix, accuracy_score
from bayes_opt import BayesianOptimization
from lightgbm import LGBMClassifier, callback
from tqdm import tqdm
import itertools

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Function for Bayesian Optimization
def lgb_eval_function(num_leaves, feature_fraction, bagging_fraction, X_train, y_train, X_test, y_test):
    params = {
        'num_leaves': int(round(num_leaves)),
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'objective': 'binary',
        'random_state': SEED,
        'verbosity': -1,
        'metric': 'auc',
    }
    
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', callbacks=[callback.early_stopping(stopping_rounds=10, first_metric_only=True)])
    pred = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, pred)


# Load dataset
# Assuming preprocessed X_train_val_resampled, y_train_val_resampled, X_test_reshaped, y_test

# Bayesian Optimization
param_bounds = {'num_leaves': (20, 40), 'feature_fraction': (0.6, 0.9), 'bagging_fraction': (0.6, 0.9)}
optimizer = BayesianOptimization(f=lambda num_leaves, feature_fraction, bagging_fraction: lgb_eval_function(num_leaves, feature_fraction, bagging_fraction, X_train_val_resampled, y_train_val_resampled, X_test_reshaped, y_test), pbounds=param_bounds, random_state=SEED)
optimizer.maximize(init_points=5, n_iter=25)
best_params = optimizer.max['params']
best_params.update({'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'force_col_wise': True, 'random_state': SEED})

# Stratified K-Fold Cross-Validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
auc_scores, models = [], []
for train_index, val_index in kf.split(X_train_val_resampled, y_train_val_resampled):
    X_train, X_val = X_train_val_resampled[train_index], X_train_val_resampled[val_index]
    y_train, y_val = y_train_val_resampled[train_index], y_train_val_resampled[val_index]
    
    model = LGBMClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[callback.early_stopping(stopping_rounds=10, first_metric_only=True)])
    
    pred_val = model.predict_proba(X_val)[:, 1]
    auc_scores.append(roc_auc_score(y_val, pred_val))
    models.append(model)
    print(f'Fold AUC: {auc_scores[-1]}')

# Save best model
best_model = models[np.argmax(auc_scores)]
joblib.dump(best_model, 'best_lgbm_model.pkl')

# Model Evaluation
print(f'Average AUC: {np.mean(auc_scores)}')
test_pred = best_model.predict(X_test_reshaped)
accuracy = accuracy_score(y_test, test_pred)
print("Accuracy: ", accuracy)
print(classification_report(y_test, test_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, test_pred)
plot_confusion_matrix(cm, ['HIGH RISK', 'LOW RISK'])

# Precision, Recall, and F1 Score Calculation
result = []
for label in [0,1]:
    precision, recall, f_score, _ = precision_recall_fscore_support(y_test == label, test_pred == label)
    result.append([label, recall[0], recall[1], precision[1], f_score[1]])
df = pd.DataFrame(result, columns=["Label", "Specificity", 'Recall', 'Precision', 'F1-Score'])
print(df)

# Bootstrap confidence interval calculation
metric_results = bootstrap_metrics(y_test, test_pred)
for metric, values in metric_results.items():
    print(f"{metric}: Mean = {values['Mean']:.4f}, 95% CI = {values['95% CI'][0]:.4f} - {values['95% CI'][1]:.4f}")
