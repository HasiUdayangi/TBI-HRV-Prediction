#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                             roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, f1_score)
# Import custom performance analysis functions
from performance_analysis import evaluate_metrics, plot_confusion_matrix, plot_roc_curve, bootstrap_metrics

# ----------------------
# Data Loading
# ----------------------
data_folder = 'Data'
X_train_val_resampled = np.load(os.path.join(data_folder, 'X_train_val_resampled.npy'))
y_train_val_resampled = np.load(os.path.join(data_folder, 'y_train_val_resampled.npy'))
X_test = np.load(os.path.join(data_folder, 'X_test.npy'))
y_test = np.load(os.path.join(data_folder, 'y_test.npy'))

# ----------------------
# Model Parameters
# ----------------------
feature_size = 26
lstm_units_options = [32, 64, 256]
learning_rate_options = [0.001]

# ----------------------
# Model Creation Function
# ----------------------
def create_model(lstm_units, learning_rate, feature_size):
    input_layer = Input(shape=(None, feature_size))
    lstm_layer = LSTM(lstm_units, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(input_layer)
    output_layer = Dense(1, activation='sigmoid')(lstm_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ----------------------
# Early Stopping Configuration
# ----------------------
early_stopping = EarlyStopping(
    monitor='val_loss',      # Monitor the validation loss
    patience=10,             # Number of epochs with no improvement after which training stops
    restore_best_weights=True  # Restore weights from the epoch with the lowest validation loss
)

# ----------------------
# Cross-Validation and Model Selection
# ----------------------
best_accuracy = 0
best_metrics = {}
best_params = {}
best_model = None
model_histories = []
model_configurations = []

# Set up StratifiedKFold for cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for units in lstm_units_options:
    for lr in learning_rate_options:
        print(f"Training with units={units} and learning_rate={lr}")
        # Create a new model instance for each configuration
        model = create_model(lstm_units=units, learning_rate=lr, feature_size=feature_size)
        fold_metrics = []
        
        for fold, (train_index, val_index) in enumerate(kf.split(X_train_val_resampled, y_train_val_resampled)):
            print(f"  Fold {fold+1}")
            X_train_fold = X_train_val_resampled[train_index]
            y_train_fold = y_train_val_resampled[train_index]
            X_val_fold = X_train_val_resampled[val_index]
            y_val_fold = y_train_val_resampled[val_index]
            
            # Fit the model on the current fold's training data
            history = model.fit(X_train_fold, y_train_fold,
                                validation_data=(X_val_fold, y_val_fold),
                                epochs=50, batch_size=32,
                                callbacks=[early_stopping], verbose=0)
            model_histories.append(history)
            model_configurations.append({'units': units, 'learning_rate': lr, 'history': history})
            
            # Evaluate performance on the validation fold using a custom function
            y_val_pred_prob = model.predict(X_val_fold)
            y_val_pred = (y_val_pred_prob > 0.5).astype(int)
            metrics = evaluate_metrics(y_val_fold, y_val_pred, y_val_pred_prob)
            fold_metrics.append(metrics)
        
        # Calculate the mean of each metric across the folds
        mean_metrics = {metric: np.nanmean([m[metric] for m in fold_metrics]) for metric in fold_metrics[0]}
        print(f"Parameters: Units={units}, LR={lr}, Metrics={mean_metrics}")
        
        # Update the best model if current mean accuracy is higher
        if mean_metrics.get('Accuracy', 0) > best_accuracy:
            best_accuracy = mean_metrics.get('Accuracy', 0)
            best_metrics = mean_metrics
            best_params = {'units': units, 'learning_rate': lr}
            best_model = tf.keras.models.clone_model(model)
            # Copy weights since clone_model only copies architecture
            best_model.set_weights(model.get_weights())

# ----------------------
# Save the Best Model
# ----------------------
if best_model:
    if not os.path.exists('MODELS'):
        os.makedirs('MODELS')
    best_model.save('MODELS/best_lstm_model.h5')
    print("Best model saved.")

print("Best Model Parameters:", best_params)
print("Best Model Performance:", best_metrics)

# ----------------------
# Evaluation on Test Set
# ----------------------
# Predict probabilities for the test set
y_test_pred_prob = best_model.predict(X_test)
y_test_pred = (y_test_pred_prob > 0.5).astype(int)

# Print classification report
print("Test Set Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=['High Risk', 'Low Risk']))

# Compute and print the confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(cm)
# Plot the confusion matrix using the custom function
plot_confusion_matrix(cm, classes=['High Risk', 'Low Risk'])

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_test_pred_prob)
roc_auc = roc_auc_score(y_test, y_test_pred_prob)
# Plot ROC curve using the custom function
plot_roc_curve(fpr, tpr, roc_auc)

# Calculate additional test metrics
test_auc = roc_auc
test_auprc = average_precision_score(y_test, y_test_pred_prob)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
test_sensitivity = tp / (tp + fn)
test_specificity = tn / (tn + fp)
test_ppv = precision_score(y_test, y_test_pred)
test_npv = tn / (tn + fn)

# Print all test metrics
print("Test Metrics:")
print(f"AUROC: {test_auc:.4f}")
print(f"AUPRC: {test_auprc:.4f}")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"Sensitivity: {test_sensitivity:.4f}")
print(f"Specificity: {test_specificity:.4f}")
print(f"PPV: {test_ppv:.4f}")
print(f"NPV: {test_npv:.4f}")

# Optionally, compute bootstrap confidence intervals (or other metrics) using the custom function
bootstrap_results = bootstrap_metrics(y_test, y_test_pred_prob, n_bootstrap=1000)
print("Bootstrap Metrics for AUC:", bootstrap_results)

