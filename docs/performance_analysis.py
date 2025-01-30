import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    """
    Function to plot a confusion matrix.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_test, y_pred_prob):
    """
    Function to plot the ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def bootstrap_metrics(y_true, y_pred, n_bootstraps=1000):
    """
    Calculate bootstrap confidence intervals for classification metrics.
    """
    np.random.seed(42)
    auc_scores = []
    accuracy_scores = []

    for _ in range(n_bootstraps):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_bootstrap = np.array(y_true)[indices]
        y_pred_bootstrap = np.array(y_pred)[indices]

        auc_scores.append(roc_auc_score(y_true_bootstrap, y_pred_bootstrap))
        accuracy_scores.append(accuracy_score(y_true_bootstrap, y_pred_bootstrap))

    return {
        'AUC': {'Mean': np.mean(auc_scores), '95% CI': (np.percentile(auc_scores, 2.5), np.percentile(auc_scores, 97.5))},
        'Accuracy': {'Mean': np.mean(accuracy_scores), '95% CI': (np.percentile(accuracy_scores, 2.5), np.percentile(accuracy_scores, 97.5))}
    }


