import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_fscore_support, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns


def load_model_and_data(model_path, test_data_path):
    """
    Load trained model and test data.

    Args:
        model_path (str): Path to saved model file.
        test_data_path (str): Path to test dataset.

    Returns:
        model: Loaded Keras model.
        X_test (np.array): Test feature data.
        y_test (np.array): Test labels.
    """
    model = load_model(model_path)
    with open(test_data_path + "/X_test.pkl", "rb") as f:
        X_test = pickle.load(f)
    with open(test_data_path + "/y_test.pkl", "rb") as f:
        y_test = pickle.load(f)

    X_test_padded = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=288, dtype='float32', padding='post', truncating='post')
    return model, X_test_padded, y_test

def get_predictions(model, X_test):
    """
    Generate model predictions.

    Args:
        model: Trained Keras model.
        X_test (np.array): Padded test sequences.

    Returns:
        y_prob (np.array): Probability scores.
        y_pred (np.array): Binary predictions.
    """
    y_prob = model.predict(X_test)[:, 0]
    y_pred = (y_prob > 0.5).astype(int)
    return y_prob, y_pred

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """
    Plot confusion matrix.

    Args:
        y_true (np.array): True labels.
        y_pred (np.array): Predicted labels.
        classes (list): Class names.
        save_path (str, optional): Path to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    if save_path:
        plt.savefig(save_path)

def compute_metrics(y_true, y_pred):
    """
    Compute model evaluation metrics.

    Args:
        y_true (np.array): True labels.
        y_pred (np.array): Predicted labels.

    Returns:
        pd.DataFrame: DataFrame with specificity, recall, precision, and F1-score.
    """
    labels = [0, 1]
    results = []
    
    for label in labels:
        precision, recall, f_score, support = precision_recall_fscore_support(np.array(y_true) == label, np.array(y_pred) == label)
        results.append([label, recall[0], recall[1], precision[1], f_score[1]])

    df_metrics = pd.DataFrame(results, columns=["Label", "Specificity", "Recall", "Precision", "F-Score"])
    return df_metrics


def bootstrap_metrics(y_true, y_pred_prob, num_samples=1000):
    """
    Compute bootstrap confidence intervals for AUC, Precision, Recall, and F1-score.

    Args:
        y_true (np.array): True labels.
        y_pred_prob (np.array): Predicted probability scores.
        num_samples (int): Number of bootstrap samples.

    Returns:
        dict: Dictionary with mean values and 95% confidence intervals.
    """
    metrics_dict = {'AUC': [], 'Precision': [], 'Recall': [], 'F1-Score': []}
    
    for _ in range(num_samples):
        sample_indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_sample_true = np.array(y_true)[sample_indices]
        y_sample_pred_prob = np.array(y_pred_prob)[sample_indices]
        y_sample_pred = (y_sample_pred_prob > 0.5).astype(int)

        # Compute metrics
        fpr, tpr, _ = roc_curve(y_sample_true, y_sample_pred_prob)
        auc_score = auc(fpr, tpr)
        precision, recall, f1, _ = precision_recall_fscore_support(y_sample_true, y_sample_pred, average='binary')

        metrics_dict['AUC'].append(auc_score)
        metrics_dict['Precision'].append(precision)
        metrics_dict['Recall'].append(recall)
        metrics_dict['F1-Score'].append(f1)

    # Compute mean and confidence intervals
    results = {}
    for metric, values in metrics_dict.items():
        mean_val = np.mean(values)
        ci_lower, ci_upper = np.percentile(values, [2.5, 97.5])
        results[metric] = {'Mean': mean_val, '95% CI': (ci_lower, ci_upper)}

    return results


def plot_precision_recall_curve(y_true, y_pred_prob, save_path=None):
    """
    Plot Precision-Recall Curve.

    Args:
        y_true (np.array): True labels.
        y_pred_prob (np.array): Predicted probabilities.
        save_path (str, optional): Path to save the plot.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, marker='o', linestyle='-', label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.show()
    if save_path:
        plt.savefig(save_path)

def save_results(df_metrics, bootstrap_results, output_folder):
    """
    Save computed metrics.

    Args:
        df_metrics (pd.DataFrame): Metrics dataframe.
        bootstrap_results (dict): Bootstrap confidence intervals.
        output_folder (str): Directory to save results.
    """
    df_metrics.to_csv(f"{output_folder}/model_test_metrics.csv", index=False)
    with open(f"{output_folder}/bootstrap_metrics.pkl", "wb") as f:
        pickle.dump(bootstrap_results, f)
        
def main():
    """
    Run model testing pipeline.
    """
    model_path = "models/best_model.h5"
    test_data_path = "data/processed"
    output_folder = "results"

    # Load model and test data
    model, X_test, y_test = load_model_and_data(model_path, test_data_path)

    # Generate predictions
    y_prob, y_pred = get_predictions(model, X_test)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, ['HIGH RISK', 'LOW RISK'], save_path=f"{output_folder}/confusion_matrix.png")

    # Compute performance metrics
    df_metrics = compute_metrics(y_test, y_pred)
    print(df_metrics)

    # Compute bootstrap metrics
    bootstrap_results = bootstrap_metrics(y_test, y_prob)
    for metric, values in bootstrap_results.items():
        print(f"{metric}: Mean = {values['Mean']:.4f}, 95% CI = {values['95% CI'][0]:.4f} - {values['95% CI'][1]:.4f}")

    # Plot precision-recall curve
    plot_precision_recall_curve(y_test, y_prob, save_path=f"{output_folder}/precision_recall_curve.png")

    # Save results
    save_results(df_metrics, bootstrap_results, output_folder)


if __name__ == "__main__":
    main()

