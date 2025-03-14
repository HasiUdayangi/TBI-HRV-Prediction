import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from utils import resample_ecg, MHTD, zero_mean_normalise

def resample_ecg(ECG, ogFS):
    fs = 240
    N = int((len(ECG) / ogFS) * fs)
    SIGNAL.resample(ECG, N)
    return ECG

def get_R_peak_windowed(ECG, fs, L=1):
    qrs_inds = MHTD(ECG, fs).astype(int)

    W = fs * L  # Window length
    qrs_inds = qrs_inds[qrs_inds > int(W / 2)]
    qrs_inds = qrs_inds[qrs_inds < (len(ECG) - int(W / 2))]
    N = len(qrs_inds)

    # Extract ECG segments centered around R-peaks
    sig = np.zeros((N, W))
    sig_data = []
    for i in range(N):
        st, en = qrs_inds[i] - int(W / 2), qrs_inds[i] + int(W / 2)
        sig[i, :] = ECG[st:en]
    sig_data.append(sig)
    sig_df = pd.DataFrame(np.concatenate(sig_data))

    # Normalize signals
    sig_norm_data = []
    for i in range(N):
        sig[i, :] = ECG[st:en]
        sig_norm = zero_mean_normalise(sig)
    sig_norm_data.append(sig_norm)
    sig_norm_df = pd.DataFrame(np.concatenate(sig_norm_data))

    return sig_df, sig_norm_df, qrs_inds, N

def detect_and_filter_ectopic_beats(df, model_ectopic_detection, fs=128):
    """
    Detects ectopic beats in ECG and removes their associated RR intervals.

    Parameters:
    - df: DataFrame, containing ECG data
    - model_path: str, Path to the pre-trained deep learning model
    - fs: int, Sampling frequency (default: 128 Hz)

    Returns:
    - filtered_rr_intervals: np.array, RR intervals after removing ectopic beats
    - ectopic_labels: list, List of ectopic beats detected (1 = ectopic, 0 = normal)
    """
    # Load the trained ectopic beat detection model
    #model = load_model(model_path)

    # Resample ECG to 128 Hz
    resampled_ecg = resample_ecg(df, ogFS=fs)
    print("ECG Resampling Complete")

    # Extract R-peak-centered ECG segments
    sig_df, sig_norm_df, qrs_inds, N = get_R_peak_windowed(resampled_ecg, fs)

    # Prepare ECG segments for model prediction
    X_test = np.expand_dims(sig_norm_df.iloc[:, :127], axis=2)
    pred_label = model_ectopic_detection.predict(X_test)
    labels = np.argmax(pred_label, axis=-1)

    # Remove RR intervals associated with ectopic beats
    rr_intervals = np.diff(qrs_inds)  # Compute RR intervals
    normal_rr_intervals = []
    
    for i in range(len(labels) - 1):
        if labels[i] == 0:  # Keep normal beats only
            normal_rr_intervals.append(rr_intervals[i])

    print(f"Ectopic Beats Detected: {sum(labels)} out of {len(labels)} beats")
    
    return np.array(normal_rr_intervals), labels.tolist()

