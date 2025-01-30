import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, coolwarm
from tqdm import tqdm
import tensorflow as tf

def calculate_feature_importance(model, X_test, y_test, feature_names, random_seed=42):
    """
    Compute permutation-based feature importance on the test dataset.

    Args:
        model: Trained LSTM model.
        X_test (np.array): Test feature data.
        y_test (np.array): Test labels.
        feature_names (list): Names of input features.
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Feature importance results.
    """
    np.random.seed(random_seed)
    
    # Initialize results list
    results = []
    
    # Compute baseline performance
    test_preds = model.predict(X_test, verbose=0).squeeze()
    baseline_mae = np.mean(np.abs(test_preds - y_test))
    results.append({'feature': 'BASELINE', 'binary_crossentropy': baseline_mae})

    # Compute importance for each feature
    for k in tqdm(range(len(feature_names)), desc="Calculating Feature Importance"):
        save_col = X_test[:, :, k].copy()
        np.random.shuffle(X_test[:, :, k])  # Shuffle one feature column
        
        test_preds = model.predict(X_test, verbose=0).squeeze()
        permuted_mae = np.mean(np.abs(test_preds - y_test))
        
        results.append({'feature': feature_names[k], 'binary_crossentropy': permuted_mae})
        X_test[:, :, k] = save_col  # Restore original feature values

    # Convert results to DataFrame
    df_importance = pd.DataFrame(results)
    df_importance['importance'] = df_importance['binary_crossentropy'] - baseline_mae  # Relative importance
    df_importance = df_importance.sort_values('binary_crossentropy')

    return df_importance


def plot_feature_importance(df_importance, save_path=None):
    """
    Plot permutation-based feature importance.

    Args:
        df_importance (pd.DataFrame): DataFrame with feature importance values.
        save_path (str, optional): Path to save the plot.
    """
    norm = Normalize(vmin=df_importance.importance.min(), vmax=df_importance.importance.max())
    sm = ScalarMappable(norm=norm, cmap=coolwarm)

    plt.figure(figsize=(10, 20))
    colors = sm.to_rgba(df_importance.importance)
    plt.barh(np.arange(len(df_importance)), df_importance.binary_crossentropy, color=colors)
    plt.yticks(np.arange(len(df_importance)), df_importance.feature.values)
    
    plt.title('LSTM Feature Importance (Test Set)', size=16)
    plt.xlabel('Binary Crossentropy with Feature Permuted', size=14)
    plt.ylabel('Feature', size=14)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ==============================
# 🚀 MAIN FUNCTION
# ==============================
def main(model, X_test, y_test, feature_names):
    """
    Run feature importance pipeline on the test set.

    Args:
        model: Trained LSTM model.
        X_test (np.array): Test feature data.
        y_test (np.array): Test labels.
        feature_names (list): Names of input features.
    """
    output_folder = "results"

    # Compute feature importance on test dataset
    df_importance = calculate_feature_importance(model, X_test, y_test, feature_names)

    # Save importance results
    df_importance.to_csv(f"{output_folder}/lstm_feature_importance_test.csv", index=False)

    # Plot importance
    plot_feature_importance(df_importance, save_path=f"{output_folder}/feature_importance_test_plot.png")

    print("Feature importance analysis on test dataset complete. Results saved in:", output_folder)



