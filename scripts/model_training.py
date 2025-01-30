import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Multiply, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, average_precision_score, accuracy_score, confusion_matrix, precision_score
import itertools


# Define parameter grid
param_dict = {
    'activation': ['sigmoid'],
    'dropout': [0.3, 0.25, 0],  # Dropout options
    'units': [16, 32, 64, 128, 256],  # LSTM Units
    'layers': [1, 2],  # Number of layers
    'optimizer': ['adam']  # Optimizer options
}

def create_model(input_shape, units, activation, dropout, optimizer, layers):
    """
    Create an LSTM-based model for sequence classification.
    
    Parameters:
    - input_shape (tuple): Shape of input sequences (timesteps, features).
    - units (int): Number of LSTM units.
    - activation (str): Activation function.
    - dropout (float): Dropout rate.
    - optimizer (str): Optimizer.
    - layers (int): Number of LSTM layers.

    Returns:
    - model (tf.keras.Model): Compiled Keras model.
    """
    inputs = Input(shape=input_shape)

    # Optional: Custom Weight Prediction Layer
    weights = WeightPredictor(input_shape[1])(inputs)
    weighted_inputs = Multiply()([inputs, weights])

    # LSTM Layers
    if layers == 1:
        lstm_out = Bidirectional(LSTM(units, return_sequences=False, kernel_regularizer=l2(0.01)))(weighted_inputs)
    else:
        lstm_out = Bidirectional(LSTM(units, return_sequences=True, kernel_regularizer=l2(0.01)))(weighted_inputs)
        lstm_out = Dropout(dropout)(lstm_out)
        lstm_out = Bidirectional(LSTM(units, return_sequences=False, kernel_regularizer=l2(0.01)))(lstm_out)

    # Fully Connected Layers
    lstm_out = Dropout(dropout)(lstm_out)
    dense_out = Dense(20, activation='relu')(lstm_out)
    outputs = Dense(1, activation='sigmoid')(dense_out)

    # Model Compilation
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', AUC()])
    
    return model


def train_and_tune_model(X_train_val_resampled, y_train_val_resampled, param_dict, num_splits=5, batch_size=32, epochs=100):
    """
    Train and tune the LSTM model using K-fold cross-validation.
    
    Parameters:
    - X_train_val_resampled (numpy array): Training data.
    - y_train_val_resampled (numpy array): Labels for training data.
    - param_dict (dict): Dictionary of hyperparameters for tuning.
    - num_splits (int): Number of cross-validation folds.
    - batch_size (int): Batch size for training.
    - epochs (int): Number of training epochs.

    Returns:
    - best_model_params (dict): Hyperparameters of the best-performing model.
    - model_histories (list): Training history of each fold.
    """
    kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    best_auc = 0
    best_model_params = {}
    model_histories = []

    # Generate all hyperparameter combinations
    param_combinations = list(itertools.product(*param_dict.values()))

    for config in param_combinations:
        config = dict(zip(param_dict.keys(), config))  # Convert tuple to dictionary
        print(f"Testing Configuration: {config}")
        fold_metrics = []

        for train_index, val_index in kf.split(X_train_val_resampled, y_train_val_resampled):
            X_train, X_val = X_train_val_resampled[train_index], X_train_val_resampled[val_index]
            y_train, y_val = y_train_val_resampled[train_index], y_train_val_resampled[val_index]

            optimizer = Adam(learning_rate=0.001, clipvalue=0.5)
            model = create_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                units=config['units'],
                activation=config['activation'],
                dropout=config['dropout'],
                optimizer=optimizer,
                layers=config['layers']
            )

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)
            model_histories.append(history)

            # Evaluate Model
            y_val_pred_prob = model.predict(X_val).ravel()
            y_val_pred = (y_val_pred_prob >= 0.5).astype(int)

            # Compute Metrics
            val_auc = auc(*roc_curve(y_val, y_val_pred_prob)[:2])
            val_auprc = average_precision_score(y_val, y_val_pred_prob)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
            val_sensitivity = tp / (tp + fn)
            val_specificity = tn / (tn + fp)
            val_ppv = precision_score(y_val, y_val_pred)
            val_npv = tn / (tn + fn)

            fold_metrics.append({
                'AUC': val_auc,
                'AUPRC': val_auprc,
                'Accuracy': val_accuracy,
                'Sensitivity': val_sensitivity,
                'Specificity': val_specificity,
                'PPV': val_ppv,
                'NPV': val_npv
            })

        # Aggregate Fold Metrics
        mean_metrics = {metric: np.mean([fm[metric] for fm in fold_metrics]) for metric in fold_metrics[0]}
        print(f"Mean Validation Metrics: {mean_metrics}")

        # Track the Best Model
        if mean_metrics['AUC'] > best_auc:
            best_auc = mean_metrics['AUC']
            best_model_params = {
                'units': config['units'],
                'activation': config['activation'],
                'dropout': config['dropout'],
                'layers': config['layers'],
                'auc': best_auc
            }
            print(f"🎯 New Best Model Found! AUC: {best_auc:.4f}")

    print(f"Best Model Parameters: {best_model_params}")
    return best_model_params, model_histories


# --------------------------
# 🚀 Execution Pipeline
# --------------------------
if __name__ == "__main__":
    # Load preprocessed data
    with open("data/processed/X_train_val_resampled.pkl", "rb") as f: X_train_val_resampled = pickle.load(f)
    with open("data/processed/y_train_val_resampled.pkl", "rb") as f: y_train_val_resampled = pickle.load(f)

    # Train and Tune Model
    best_model_params, model_histories = train_and_tune_model(X_train_val_resampled, y_train_val_resampled, param_dict)

