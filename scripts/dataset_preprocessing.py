import os
import pandas as pd
import numpy as np
import pickle
import boto3
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from imblearn.over_sampling import SMOTE


s3_client = boto3.client('s3')
bucket_name = ""
hrv_folder_gcuh = ""
hrv_folder_incart = ""

def fetch_csv_files_from_folder(folder_prefix, source_label):
    """
    Fetch list of CSV files from a given folder in the S3 bucket,
    and add a 'source' column with the specified label.
    Returns a list of DataFrames.
    """
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix)
    files = response.get('Contents', [])
    dfs = []
    for file in files:
        file_key = file['Key']
        try:
            obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            df = pd.read_csv(obj['Body'])
            df['source'] = source_label  # Tag the DataFrame with its source
            dfs.append(df)
        except Exception as e:
            print(f"Error fetching {file_key}: {e}")
    return dfs
    
def fetch_and_merge_hrv_data():

    dfs_gcuh = fetch_csv_files_from_folder(hrv_folder_gcuh, "GCUH")
    dfs_incart = fetch_csv_files_from_folder(hrv_folder_incart, "INCART")
    all_dfs = dfs_gcuh + dfs_incart
    if all_dfs:
        combined_data = pd.concat(all_dfs, ignore_index=True)
        print(f" Merged {len(all_dfs)} HRV CSV files into one DataFrame.")
    else:
        combined_data = pd.DataFrame()
        print("No HRV data files found.")
    return combined_data



def create_sequences_labels(combined_data):
    sequences = []
    labels = []
    sources = []
    mortality_col = "mortality" if "mortality" in combined_data.columns else "motality"
    
    for pid, group in combined_data.groupby('patient_id'):
        patient_sequences = group.drop(columns=['patient_id', mortality_col, 'source']).values
        sequences.append(patient_sequences)
        labels.append(group[mortality_col].iloc[0])
        sources.append(group['source'].iloc[0])
    print(f" Created sequences for {len(sequences)} patients.")
    return sequences, np.array(labels), sources

def split_and_apply_smote_by_source(sequences, labels, sources, output_dir="data/processed"):
    """
    For each source (GCUH and INCART), split patient data into 80% training and 20% testing sets,
    pad the sequences, apply SMOTE to each source's training set separately, then combine the results.
    Also, combine the testing sets from both sources.
    
    Returns:
      X_train_combined (numpy array): Combined SMOTE-resampled training data (3D array).
      y_train_combined (numpy array): Combined training labels.
      X_test_combined (numpy array): Combined testing data (padded, 3D array).
      y_test_combined (numpy array): Combined testing labels.
    """
    import pickle  # Ensure pickle is imported here
    X_train_all = []
    y_train_all = []
    X_test_all = []
    y_test_all = []
    
    unique_sources = np.unique(sources)
    max_timesteps = 288  # e.g., 24 hours of 5-minute segments
    for src in unique_sources:
        # Get indices for the current source
        idx = [i for i, s in enumerate(sources) if s == src]
        src_sequences = [sequences[i] for i in idx]
        src_labels = labels[idx]
        
        # Split into 80% train, 20% test for this source
        X_train, X_test, y_train, y_test = train_test_split(
            src_sequences, src_labels, test_size=0.2, random_state=42, stratify=src_labels)
        print(f"{src}: {len(X_train)} training patients, {len(X_test)} testing patients.")
        
        # Pad sequences for training and testing separately
        X_train_padded = pad_sequences(X_train, maxlen=max_timesteps, dtype='float32', 
                                       padding='post', truncating='post')
        X_test_padded = pad_sequences(X_test, maxlen=max_timesteps, dtype='float32', 
                                      padding='post', truncating='post')
        
        # Flatten training set for SMOTE application (2D array)
        X_train_flat = X_train_padded.reshape(X_train_padded.shape[0], -1)
        
        # Apply SMOTE separately on this source's training data
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, y_train)
        
        # Reshape back to 3D: (num_patients, timesteps, feature_dim)
        feature_dim = X_train_resampled.shape[1] // max_timesteps
        X_train_resampled_3d = X_train_resampled.reshape(-1, max_timesteps, feature_dim)
        
        # Append the results to the overall lists
        X_train_all.append(X_train_resampled_3d)
        y_train_all.append(y_train_resampled)
        X_test_all.append(X_test_padded)  # Testing sets are padded but not resampled
        y_test_all.append(y_test)
    
    # Combine training sets from all sources
    X_train_val_resampled= np.concatenate(X_train_all, axis=0)
    y_train_val_resampled = np.concatenate(y_train_all, axis=0)
    # Combine testing sets from all sources
    X_test = np.concatenate(X_test_all, axis=0)
    y_test = np.concatenate(y_test_all, axis=0)
    
    # Save combined sets (optional)
    with open(os.path.join(output_dir, "X_train_val_resampled.pkl"), "wb") as f:
        pickle.dump(X_train_combined, f)
    with open(os.path.join(output_dir, "y_train_val_resampled.pkl"), "wb") as f:
        pickle.dump(y_train_combined, f)
    with open(os.path.join(output_dir, "X_test.pkl"), "wb") as f:
        pickle.dump(X_test_combined, f)
    with open(os.path.join(output_dir, "y_test.pkl"), "wb") as f:
        pickle.dump(y_test_combined, f)
    
    print(" Combined training and testing sets created and saved.")
    return X_train_val_resampled, y_train_val_resampled, X_test, y_test


import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from imblearn.over_sampling import SMOTE
import os

def split_and_apply_new_smote_by_source(sequences, labels, sources, output_dir="data/processed"):

    
    X_train_all = []
    y_train_all = []
    X_test_all = []
    y_test_all = []
    
    unique_sources = np.unique(sources)
    max_timesteps = 288  # e.g., 24 hours of 5-minute segments
    for src in unique_sources:
        # Get indices for the current source
        idx = [i for i, s in enumerate(sources) if s == src]
        src_sequences = [sequences[i] for i in idx]
        src_labels = labels[idx]
        
        # Split into 80% train, 20% test for this source
        X_train, X_test, y_train, y_test = train_test_split(
            src_sequences, src_labels, test_size=0.2, random_state=42, stratify=src_labels)
        print(f"{src}: {len(X_train)} training patients, {len(X_test)} testing patients.")
 
        X_train_padded = pad_sequences(X_train, maxlen=max_timesteps, dtype='float32', 
                                       padding='post', truncating='post')
        X_test_padded = pad_sequences(X_test, maxlen=max_timesteps, dtype='float32', 
                                      padding='post', truncating='post')
        
  
        num_samples, num_timesteps, num_features = X_train_padded.shape
        X_train_flat = X_train_padded.reshape(num_samples, -1)  # Flatten for SMOTE
        
        # Apply SMOTE separately on this source's training data
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, y_train)
        
        # Reshape back to 3D: (num_patients, timesteps, feature_dim)
        feature_dim = X_train_resampled.shape[1] // max_timesteps
        X_train_resampled_3d = X_train_resampled.reshape(-1, max_timesteps, feature_dim)
        
        # Append the results to the overall lists
        X_train_all.append(X_train_resampled_3d)
        y_train_all.append(y_train_resampled)
        X_test_all.append(X_test_padded)  # Testing sets are padded but not resampled
        y_test_all.append(y_test)
    
    # Combine training sets from all sources
    X_train_val_resampled = np.concatenate(X_train_all, axis=0)
    y_train_val_resampled = np.concatenate(y_train_all, axis=0)
    # Combine testing sets from all sources
    X_test_combined = np.concatenate(X_test_all, axis=0)
    y_test_combined = np.concatenate(y_test_all, axis=0)
    
    # Save combined sets (optional)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "X_train_val_resampled.pkl"), "wb") as f:
        pickle.dump(X_train_val_resampled, f)
    with open(os.path.join(output_dir, "y_train_val_resampled.pkl"), "wb") as f:
        pickle.dump(y_train_val_resampled, f)
    with open(os.path.join(output_dir, "X_test_combined.pkl"), "wb") as f:
        pickle.dump(X_test_combined, f)
    with open(os.path.join(output_dir, "y_test_combined.pkl"), "wb") as f:
        pickle.dump(y_test_combined, f)
    
    print(" Combined training and testing sets created and saved.")
    return X_train_val_resampled, y_train_val_resampled, X_test_combined, y_test_combined


# --------------------------
# 🚀 Execution Pipeline
# --------------------------
if __name__ == "__main__":
    # 1. Fetch and merge HRV data
    combined_data = fetch_and_merge_hrv_data()

    # 2. Create sequences and labels
    sequences, labels, sources = create_sequences_labels(combined_data)

    split_and_apply_new_smote_by_source(sequences, labels, sources)

