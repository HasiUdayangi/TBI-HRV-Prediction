#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Masking, Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, accuracy_score, precision_score, f1_score

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
# Hyperparameters
# ----------------------
gru_units_options = [32, 64, 128]        # You can adjust these as needed
learning_rate_options = [0.001, 0.0005]   # You can adjust these as needed

# ----------------------
# Create BiGRU Model Function
# ----------------------
def create_model(input_shape, gru_units, learning_rate):
    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.0)(inputs)
    gru_out = Bidirectional(GRU(gru_units, return_sequences=False, kernel_regularizer=l2(0.01)))(masked_inputs)
    gru_out = Dropout(0.3)(gru_out)
    full1 = Dense(20, activation='relu')(gru_out)
    outputs = Dense(1, activation='sigmoid')(full1)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ----------------------
# Cross-Validation Setup
# ----------------------
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
best_accuracy = 0.0
best_model = None
best_params = {}
auc_scores = []
model_histories = []
models_list = []

# ----------------------
# Hyperparameter Tuning & Cross-Validation
# ----------------------
fold_global = 1  # global fold counter across hyperparameter combinations

for gru_units in gru_units_options:
    for lr in learning_rate_options:
        print(f"\nTraining with GRU Units: {gru_units}, Learning Rate: {lr}")
        fold_auc_scores = []
        # For each hyperparameter combination, run cross-validation
        for train_index, val_index in skf.split(X_train_val_resampled, y_train_val_resampled):
            X_train = X_train_val_resampled[train_index]
            y_train = y_train_val_resampled[train_index]
            X_val = X_train_val_resampled[val_index]
            y_val = y_train_val_resampled[val_index]
            
            model = create_model((X_train.shape[1], X_train.shape[2]), gru_units, lr)
            
            history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                                validation_data=(X_val, y_val), verbose=1,
                                callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
            model_histories.append(history)
            
            # Evaluate on the validation set
            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            print(f"Fold {fold_global} - Validation Accuracy: {val_accuracy:.4f}, Loss: {val_loss:.4f}")
            
            # Save best model if current fold's accuracy is the highest seen so far
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model
                best_params = {'gru_units': gru_units, 'learning_rate': lr}
                # Save the best model to file
                model.save('best_model_weight_BiGRU.h5')
                print(f"New best model saved with Validation Accuracy: {val_accuracy:.4f}")
            
            # Predict probabilities on validation set and calculate AUC
            y_val_pred_prob = model.predict(X_val).ravel()
            val_auc = roc_auc_score(y_val, y_val_pred_prob)
            fold_auc_scores.append(val_auc)
            
            # Optionally, compute additional metrics using evaluate_metrics
            metrics = evaluate_metrics(y_val, (y_val_pred_prob >= 0.5).astype(int), y_val_pred_prob)
            print(f"Fold {fold_global} Metrics: {metrics}")
            fold_global += 1
        
        mean_auc = np.mean(fold_auc_scores)
        auc_scores.append(mean_auc)
        print(f"Parameters: GRU Units: {gru_units}, Learning Rate: {lr}, Mean AUC across folds: {mean_auc:.4f}")

# ----------------------
# Overall Best Model Evaluation on Test Set
# ----------------------
if best_model:
    print("\nBest Model Hyperparameters:", best_params)
    # Evaluate best model on test set
    y_test_pred_prob = best_model.predict(X_test).ravel()
    y_test_pred = (y_test_pred_prob >= 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_pred_prob)
    test_report = classification_report(y_test, y_test_pred, target_names=['High Risk', 'Low Risk'])
    print("\nTest Set Evaluation:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"AUC: {test_auc:.4f}")
    print("Classification Report:")
    print(test_report)
    
    # Compute and display confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:")
    print(cm)
    plot_confusion_matrix(cm, classes=['High Risk', 'Low Risk'])
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_prob)
    plot_roc_curve(fpr, tpr, test_auc)
    
    # Optionally, compute bootstrap confidence intervals for AUC
    bootstrap_results = bootstrap_metrics(y_test, y_test_pred_prob, n_bootstrap=1000)
    print("Bootstrap Metrics for AUC:", bootstrap_results)
else:
    print("No best model found during cross-validation.")


