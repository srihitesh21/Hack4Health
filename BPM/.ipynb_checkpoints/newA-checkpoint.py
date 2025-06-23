import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, periodogram

def run_pulse_rate_from_csv(csv_path, fs=0.5):
    """
    Process PPG data from Arduino CSV and estimate heart rate.

    Parameters:
        csv_path (str): Path to CSV containing 'IR_Value'
        fs (float): Sampling frequency in Hz (1 sample per 2 seconds = 0.5 Hz)

    Returns:
        Tuple (bpm_estimate, confidence)
    """
    # Load the CSV
    df = pd.read_csv(csv_path, skiprows=1)
    ppg_signal = df["IR_Value"].astype(float).values

    # Bandpass filter (40–240 BPM = 0.67–4 Hz)
    low, high = 0.67, 4.0
    nyq = 0.5 * fs
    b, a = butter(2, [low / nyq, high / nyq], btype="band")
    filtered = filtfilt(b, a, ppg_signal)

    # Power spectrum
    freqs, power = periodogram(filtered, fs)
    
    # Limit to heart rate band
    mask = (freqs >= low) & (freqs <= high)
    freqs, power = freqs[mask], power[mask]

    # Find peak frequency and convert to BPM
    peak_freq = freqs[np.argmax(power)]
    bpm = peak_freq * 60
    confidence = power.max() / np.sum(power)  # simple confidence estimate

    return bpm, confidence

# Example usage
if __name__ == "__main__":
    bpm, conf = run_pulse_rate_from_csv("arduino_data_20250621_155645.csv", fs=10)
    print(f"Estimated BPM: {bpm:.2f} | Confidence: {conf:.2f}")
