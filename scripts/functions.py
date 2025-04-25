from sklearn.metrics import roc_curve, auc, average_precision_score, accuracy_score, confusion_matrix, precision_score, recall_score
from scipy.stats import t
import numpy as np
from sklearn.utils import resample
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from inspect import signature
from sklearn.metrics import roc_curve, auc, average_precision_score, accuracy_score, confusion_matrix, precision_score, recall_score
from scipy.stats import t
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve, auc





def bootstrap_metrics(y_true, y_pred_prob, n_bootstraps=1000):
    """
    Compute the 95% CI for classification metrics using bootstrap resampling.

    Args:
        y_true (array): True binary labels.
        y_pred_prob (array): Predicted probabilities for the positive class.
        n_bootstraps (int): Number of bootstrap samples to use.

    Returns:
        dict: Dictionary of metrics with their mean and 95% confidence intervals.
    """
    # Initial predictions to binary using a threshold of 0.5
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Function to calculate metrics
    def compute_metrics(y_true, y_scores, y_pred):
        metrics = {
            'AUC': auc(*roc_curve(y_true, y_scores)[:2]),
            'AUPRC': average_precision_score(y_true, y_scores),
            'Accuracy': accuracy_score(y_true, y_pred),
            'Sensitivity': recall_score(y_true, y_pred),
            'Specificity': recall_score(y_true, y_pred, pos_label=0),
            'PPV': precision_score(y_true, y_pred, zero_division=0),
            'NPV': precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        }
        return metrics

    # Compute initial metrics
    initial_metrics = compute_metrics(y_true, y_pred_prob, y_pred)

    # Bootstrap to estimate 95% CI
    bootstrap_metrics = {key: [] for key in initial_metrics.keys()}
    for _ in range(n_bootstraps):
        # Bootstrap by sampling with replacement on the indices
        indices = resample(np.arange(len(y_true)), replace=True)
        bs_y_true = y_true[indices]
        bs_y_pred = y_pred[indices]
        bs_y_pred_prob = y_pred_prob[indices]
        
        # Calculate metrics
        bs_metrics = compute_metrics(bs_y_true, bs_y_pred_prob, bs_y_pred)
        for key in bootstrap_metrics:
            bootstrap_metrics[key].append(bs_metrics[key])

    # Calculate mean and 95% confidence intervals
    metrics_mean_ci = {}
    for key, values in bootstrap_metrics.items():
        mean = np.mean(values)
        lower = np.percentile(values, 2.5)
        upper = np.percentile(values, 97.5)
        metrics_mean_ci[key] = {'Mean': mean, '95% CI': (lower, upper)}

    return metrics_mean_ci


def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",verticalalignment="top",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    #plt.savefig('hyperas-confusion-matrix.png')


def call_precision_recall_curve(truelabel,predictedlabel,actualprediction,label):
    
    #print(truelabel.shape)
    #print(actualprediction.shape)
    
    precision, recall, _ = precision_recall_curve(truelabel,actualprediction)
    au = auc(recall, precision)
    print("Area under graph "+str(au))
    print("-------------------------------------------------")
    rec_score = recall_score(truelabel,predictedlabel,pos_label=1,average='binary')
    precise_score = precision_score(truelabel,predictedlabel,pos_label=1,average='binary')
    f1 = f1_score(truelabel,predictedlabel,pos_label=1, average='binary')
    print("Precision for good review "+str(precise_score))
    print("Recall for good review "+str(rec_score))
    print("F1-score for good review  " + str(f1))
    print("-------------------------------------------------")
    rec_score = recall_score(truelabel,predictedlabel,pos_label=0,average='binary')
    precise_score = precision_score(truelabel,predictedlabel,pos_label=0,average='binary')
    f1 = f1_score(truelabel,predictedlabel,pos_label=0, average='binary')
    print("Precision for bad review "+str(precise_score))
    print("Recall for bad review "+str(rec_score))
    print("F1-score for bad review  " + str(f1))
    
    
    #plot the no-skill line too
    positive_cases = sum(truelabel)/len(truelabel)
    plt.plot([0, 1], [positive_cases, positive_cases], linestyle='--')
    
    
    
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    #plt.savefig('Precision-Recall-Curve-hyperas')



def evaluate_model(y_true, y_pred, y_pred_prob, threshold=0.5):
    """
    Evaluate and plot the performance of a binary classifier.
    
    Args:
        y_true (array-like): True binary labels.
        y_pred_prob (array-like): Predicted probabilities for the positive class.
        threshold (float): Threshold for classifying probability as positive class. Default is 0.5.
    
    Returns:
        None
    """
    # Convert probabilities to binary predictions based on the threshold
    #y_pred = (y_pred_prob > threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred_prob)
    report = classification_report(y_true, y_pred, output_dict=False)

    # Print performance metrics
    print("Accuracy:", accuracy)
    print("AUC Score:", auc_score)
    print("Classification Report:\n", report)

    # Calculate AUC ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plot AUC ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Example usage
# Assuming y_train_val and y_val_pred_prob are defined as your true labels and predicted probabilities
# evaluate_model(y_train_val, y_val_pred_prob)



def evaluate_and_plot_roc(y_true, y_pred, y_pred_prob):
    """
    Evaluate the classifier and plot ROC curves for both the positive class and
    the negative class treated as positive.
    
    Args:
        y_true (array-like): True binary labels (0 or 1).
        y_pred_prob (array-like): Predicted probabilities for the positive class.
    """
    # Convert predicted probabilities to binary predictions using a threshold of 0.5
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred_prob)
    report = classification_report(y_true, y_pred, output_dict=False)

    # Print performance metrics
    print("Accuracy:", accuracy)
    print("AUC Score:", auc_score)
    print("Classification Report:\n", report)

    # Calculate AUC ROC curve for the positive class
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Calculate AUC ROC curve for the "negative class treated as positive"
    reversed_labels = 1 - y_true  # Flipping 0s to 1s and 1s to 0s
    fpr_neg, tpr_neg, thresholds_neg = roc_curve(reversed_labels, 1 - y_pred_prob)
    roc_auc_neg = auc(fpr_neg, tpr_neg)

    # Plot AUC ROC for both classes
    plt.figure(figsize=(12, 6))

    # Subplot for the positive class
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve for positive class (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Positive Class')
    plt.legend(loc="lower right")

    # Subplot for the "negative class treated as positive"
    plt.subplot(1, 2, 2)
    plt.plot(fpr_neg, tpr_neg, color='darkgreen', lw=2, label=f'ROC curve for "negative class as positive" (area = {roc_auc_neg:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Negative Class Considered as Positive')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming y_train_val and y_val_pred_prob are your actual labels and predicted probabilities, respectively
# evaluate_and_plot_roc(y_train_val, y_val_pred_prob)
