import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, periodogram, spectrogram
import matplotlib.pyplot as plt
import matplotlib as mpl

def run_pulse_rate_from_csv(csv_path, fs=0.5, plot_spectrogram=True):
    """
    Process PPG data from Arduino CSV and estimate heart rate.

    Parameters:
        csv_path (str): Path to CSV containing 'IR_Value'
        fs (float): Sampling frequency in Hz (1 sample per 2 seconds = 0.5 Hz)
        plot_spectrogram (bool): Whether to generate and display spectrogram

    Returns:
        Tuple (bpm_estimate, confidence)
    """
    # Load the CSV
    df = pd.read_csv(csv_path, skiprows=1)
    ppg_signal = df["IR_Value"].astype(float).values
    
    print(f"Signal length: {len(ppg_signal)} samples")
    print(f"Signal duration: {len(ppg_signal)/fs:.2f} seconds")
    
    # Check if signal is too short for filtering
    if len(ppg_signal) < 10:
        print("Warning: Signal is very short. Using simple frequency analysis.")
        # Use simple periodogram approach for short signals
        freqs, power = periodogram(ppg_signal, fs)
        
        # Limit to heart rate band (40-240 BPM = 0.67-4 Hz)
        mask = (freqs >= 0.67) & (freqs <= 4.0)
        if np.any(mask):
            freqs, power = freqs[mask], power[mask]
            peak_freq = freqs[np.argmax(power)]
            bpm = peak_freq * 60
            confidence = power.max() / np.sum(power)
        else:
            bpm = 0
            confidence = 0
            
        if plot_spectrogram:
            # Simple plot for short signals
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            time_axis = np.arange(len(ppg_signal)) / fs
            plt.plot(time_axis, ppg_signal, 'b-o', label='Raw PPG Signal')
            plt.xlabel('Time (seconds)')
            plt.ylabel('IR Value')
            plt.title(f'Raw PPG Signal - Estimated BPM: {bpm:.1f}, Confidence: {confidence:.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            freqs_full, power_full = periodogram(ppg_signal, fs)
            plt.plot(freqs_full * 60, power_full, 'g-', label='Power Spectrum')
            plt.xlabel('Frequency (BPM)')
            plt.ylabel('Power')
            plt.title('Power Spectral Density')
            plt.axvline(x=bpm, color='red', linestyle='--', alpha=0.8, label=f'Estimated BPM: {bpm:.1f}')
            plt.xlim(0, 300)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('short_signal_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved: short_signal_analysis.png")
            
        return bpm, confidence
    
    # For longer signals, use bandpass filtering
    # Bandpass filter (40–240 BPM = 0.67–4 Hz)
    low, high = 0.67, 4.0
    nyq = 0.5 * fs
    
    # Adjust filter order for short signals
    filter_order = min(2, len(ppg_signal) // 3 - 1)
    if filter_order < 1:
        filter_order = 1
    
    try:
        b, a = butter(filter_order, [low / nyq, high / nyq], btype="band")
        filtered = filtfilt(b, a, ppg_signal)
    except ValueError as e:
        print(f"Filtering failed: {e}. Using unfiltered signal.")
        filtered = ppg_signal

    # Generate spectrogram similar to main algorithm
    if plot_spectrogram and len(filtered) > 8:
        # Calculate window and overlap parameters
        window_size = min(len(filtered), int(fs * 8))  # 8 second window
        overlap_size = int(window_size * 0.75)  # 75% overlap
        
        # Ensure minimum window size
        if window_size < 4:
            window_size = 4
            overlap_size = 2
        
        # 1. Matplotlib's plt.specgram (filtered signal)
        plt.figure(figsize=(12, 8))
        spec, freqs, times, im = plt.specgram(
            filtered, 
            NFFT=window_size, 
            Fs=fs, 
            noverlap=overlap_size,
            cmap='viridis'
        )
        freqs_bpm = freqs * 60
        peak_freq = freqs[np.argmax(np.mean(spec, axis=1))]
        bpm = peak_freq * 60
        confidence = np.max(np.mean(spec, axis=1)) / np.sum(np.mean(spec, axis=1))
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (BPM)')
        plt.title(f'PPG Signal Spectrogram (plt.specgram, Filtered) - Estimated BPM: {bpm:.1f}, Confidence: {confidence:.3f}')
        plt.colorbar(im, label='Power Spectral Density')
        plt.axhline(y=bpm, color='red', linestyle='--', alpha=0.8, label=f'Estimated BPM: {bpm:.1f}')
        plt.legend()
        plt.ylim(40, 240)
        plt.tight_layout()
        plt.savefig('plt_specgram_filtered.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: plt_specgram_filtered.png")
        
        # Also create a time-domain plot
        plt.figure(figsize=(12, 6))
        time_axis = np.arange(len(ppg_signal)) / fs
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, ppg_signal, 'b-', alpha=0.7, label='Raw PPG Signal')
        plt.xlabel('Time (seconds)')
        plt.ylabel('IR Value')
        plt.title('Raw PPG Signal from Arduino')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 1, 2)
        plt.plot(time_axis, filtered, 'g-', alpha=0.7, label='Filtered PPG Signal')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Filtered Value')
        plt.title('Band-pass Filtered PPG Signal (40-240 BPM)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('time_domain_signals.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: time_domain_signals.png")
        
        return bpm, confidence
    
    else:
        # Original simple approach without spectrogram
        freqs, power = periodogram(filtered, fs)
        mask = (freqs >= low) & (freqs <= high)
        freqs, power = freqs[mask], power[mask]
        if len(freqs) > 0:
            peak_freq = freqs[np.argmax(power)]
            bpm = peak_freq * 60
            confidence = power.max() / np.sum(power)
        else:
            bpm = 0
            confidence = 0
        return bpm, confidence

# Main execution - only analyze the specific Arduino file
if __name__ == "__main__":
    target_file = "A.csv"
    print("=" * 60)
    print(f"Pulse Rate Analysis for: {target_file}")
    print("=" * 60)
    try:
        bpm, conf = run_pulse_rate_from_csv(target_file, fs=10, plot_spectrogram=True)
        print(f"\nResults:")
        print(f"Estimated BPM: {bpm:.2f}")
        print(f"Confidence: {conf:.3f}")
        if bpm > 0:
            print(f"Status: Valid heart rate detected")
        else:
            print(f"Status: No clear heart rate signal detected")
    except Exception as e:
        print(f"Error analyzing file: {e}")
        print("Trying without spectrogram...")
        try:
            bpm, conf = run_pulse_rate_from_csv(target_file, fs=10, plot_spectrogram=False)
            print(f"\nResults (simple analysis):")
            print(f"Estimated BPM: {bpm:.2f}")
            print(f"Confidence: {conf:.3f}")
        except Exception as e2:
            print(f"Analysis failed: {e2}")
    print("\n" + "=" * 60)
