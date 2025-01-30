import os
import pandas as pd
import numpy as np
import pickle
import boto3
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from imblearn.over_sampling import SMOTE

# AWS S3 Configuration (Ensure your AWS credentials are configured)
s3_client = boto3.client('s3')
bucket_name = "your-s3-bucket-name"
hrv_folder = "hrv-data-folder"  # Path to HRV CSV files in the S3 bucket

def fetch_and_merge_hrv_data():
    """
    Fetch all HRV calculation CSV files from S3 and merge them into a single DataFrame.
    
    Returns:
    - combined_data (DataFrame): Merged HRV dataset
    """
    csv_files = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=hrv_folder)['Contents']
    
    dataframes = []
    for file in tqdm(csv_files, desc="Fetching HRV Data"):
        file_key = file['Key']
        obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        df = pd.read_csv(obj['Body'])
        dataframes.append(df)

    # Merge all data into one DataFrame
    combined_data = pd.concat(dataframes, ignore_index=True)
    print(f"✅ Merged {len(dataframes)} HRV CSV files into one DataFrame.")

    return combined_data

def create_sequences_labels(combined_data):
    """
    Create sequences of HRV data per patient and assign mortality labels.

    Parameters:
    - combined_data (DataFrame): Merged HRV dataset

    Returns:
    - sequences (numpy array): HRV feature sequences for model input
    - labels (numpy array): Corresponding mortality labels
    """
    sequences = []
    labels = []
    
    for pid, group in combined_data.groupby('patient_id'):
        patient_sequences = group.drop(columns=['SDANN (ms)', 'patient_id', 'patientid', 'motality', 'mortality']).values
        sequences.append(patient_sequences)
        labels.append(group['mortality'].iloc[0])  # Assign same label to all segments of a patient

    sequences = np.array(sequences, dtype=object)
    labels = np.array(labels)
    
    print(f"✅ Created {len(sequences)} patient sequences for training.")
    return sequences, labels

def split_and_save_data(sequences, labels, output_dir="data/processed"):
    """
    Split the dataset into train/test sets, pad sequences, and save as PKL files.

    Parameters:
    - sequences (numpy array): HRV feature sequences
    - labels (numpy array): Corresponding mortality labels
    - output_dir (str): Path to save processed data files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Split into training and testing sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42, stratify=labels)

    # Pad sequences to have a consistent shape
    max_timesteps = 288  # Assuming 24 hours of 5-min segments
    X_train_val_padded = pad_sequences(X_train_val, maxlen=max_timesteps, dtype='float32', padding='post', truncating='post')
    X_test_padded = pad_sequences(X_test, maxlen=max_timesteps, dtype='float32', padding='post', truncating='post')

    # Reshape for model input
    X_train_val_reshaped = X_train_val_padded.reshape(X_train_val_padded.shape[0], -1)
    X_test_reshaped = X_test_padded.reshape(X_test_padded.shape[0], -1)

    # Save preprocessed data
    with open(os.path.join(output_dir, "X_train_val.pkl"), "wb") as f: pickle.dump(X_train_val_reshaped, f)
    with open(os.path.join(output_dir, "X_test.pkl"), "wb") as f: pickle.dump(X_test_reshaped, f)
    with open(os.path.join(output_dir, "y_train_val.pkl"), "wb") as f: pickle.dump(y_train_val, f)
    with open(os.path.join(output_dir, "y_test.pkl"), "wb") as f: pickle.dump(y_test, f)

    print("✅ Train/test split complete. Processed files saved.")

def apply_smote_and_save(output_dir="data/processed"):
    """
    Apply SMOTE to balance the training dataset and reshape back to LSTM format.

    Parameters:
    - output_dir (str): Path to save resampled data
    """
    # Load data
    with open(os.path.join(output_dir, "X_train_val.pkl"), "rb") as f: X_train_val_reshaped = pickle.load(f)
    with open(os.path.join(output_dir, "y_train_val.pkl"), "rb") as f: y_train_val = pickle.load(f)

    # Apply SMOTE for class balancing
    smote = SMOTE(random_state=42)
    X_train_val_resampled, y_train_val_resampled = smote.fit_resample(X_train_val_reshaped, y_train_val)

    # Reshape back to 3D format for LSTM
    max_timesteps = 288
    feature_dim = X_train_val_resampled.shape[1] // max_timesteps
    X_train_val_resampled = X_train_val_resampled.reshape(-1, max_timesteps, feature_dim)

    # Save resampled data
    with open(os.path.join(output_dir, "X_train_val_resampled.pkl"), "wb") as f: pickle.dump(X_train_val_resampled, f)
    with open(os.path.join(output_dir, "y_train_val_resampled.pkl"), "wb") as f: pickle.dump(y_train_val_resampled, f)

    print(f"✅ SMOTE applied. New shape: {X_train_val_resampled.shape}. Resampled data saved.")

# --------------------------
# 🚀 Execution Pipeline
# --------------------------
if __name__ == "__main__":
    # 1. Fetch and merge HRV data
    combined_data = fetch_and_merge_hrv_data()

    # 2. Create sequences and labels
    sequences, labels = create_sequences_labels(combined_data)

    # 3. Split and save train/test data
    split_and_save_data(sequences, labels)

    # 4. Apply SMOTE and save balanced dataset
    apply_smote_and_save()

