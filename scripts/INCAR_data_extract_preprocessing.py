import wfdb
import resampy
import numpy as np

def process_incart_record(record_name, source_dir="charisdb", original_fs=50, target_fs=128, segment_duration=5):
    """
    Extracts, truncates, segments, and resamples ECG data from an INCART record.

    Parameters:
    - record_name (str): Name of the record (e.g., 'charis12')
    - source_dir (str): Directory where INCART records are stored.
    - original_fs (int): Original sampling frequency (default=50 Hz).
    - target_fs (int): Target resampling frequency (default=128 Hz).
    - segment_duration (int): Duration of each segment in minutes (default=5 min).

    Returns:
    - resampled_segments (list): List of 5-minute ECG segments resampled to 128 Hz.
    """
    
    # Load ECG signal
    signal, fields = wfdb.rdsamp(record_name, pn_dir=source_dir)
    fs = fields['fs']
    
    print(f"✅ Loaded {record_name} - Original Sampling Rate: {fs} Hz")

    # Calculate total number of samples for 24 hours
    samples_24_hours = 24 * 60 * 60 * fs  
    samples_5_min = segment_duration * 60 * fs  

    # Truncate to 24 hours if the signal is longer
    if len(signal) > samples_24_hours:
        signal = signal[:samples_24_hours]

    # Segment the data into 5-minute intervals
    segments = [
        signal[i:i + samples_5_min] 
        for i in range(0, len(signal), samples_5_min) 
        if len(signal[i:i + samples_5_min]) == samples_5_min
    ]

    print(f"🔹 {len(segments)} segments created (each {segment_duration} minutes)")

    # Resample each 5-minute segment to 128 Hz
    resampled_segments = [resampy.resample(seg[:, 0], original_fs, target_fs) for seg in segments]

    print(f"✅ Resampled all segments to {target_fs} Hz")

    return resampled_segments

# Example Usage:
# process_incart_record("charis12")

