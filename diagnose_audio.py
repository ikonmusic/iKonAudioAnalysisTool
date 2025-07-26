import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa.display

def diagnose_audio(file_path):
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)

    if y.size == 0:
        raise ValueError("Audio file is empty or unreadable.")

    duration = librosa.get_duration(y=y, sr=sr)

    # Volume (RMS)
    rms_vals = librosa.feature.rms(y=y)
    if rms_vals is None or rms_vals.size == 0:
        rms = 0.0
        silence_pct = 100.0
    else:
        rms = float(np.mean(rms_vals))
        rms_flat = rms_vals.flatten()
        silence_thresh = 0.02
        silence_frames = np.sum(rms_flat < silence_thresh)
        silence_pct = (silence_frames / len(rms_flat)) * 100

    # Clipping detection
    clip_thresh = 0.98
    clipped_samples = np.sum(np.abs(y) > clip_thresh)
    clipping_pct = (clipped_samples / len(y)) * 100

    # Pitch detection
    try:
        pitch = librosa.yin(y, fmin=50, fmax=300)
        avg_pitch = float(np.mean(pitch)) if pitch.size > 0 else float("nan")
    except:
        avg_pitch = float("nan")

    # Frequency energy bands (EQ snapshot)
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    def band_energy(freq_low, freq_high):
        band = stft[(freqs >= freq_low) & (freqs < freq_high)]
        return float(np.mean(band)) if band.size > 0 else 0.0

    low_energy = band_energy(20, 250)
    mid_energy = band_energy(250, 4000)
    high_energy = band_energy(4000, 16000)

    # Generate waveform preview
    waveform_path = os.path.splitext(file_path)[0] + "_waveform.png"
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, alpha=0.7)
    plt.title("Waveform Preview")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(waveform_path)
    plt.close()

    return {
        "Duration (sec)": round(duration, 2),
        "Average Volume (RMS)": round(rms, 6),
        "Average Pitch (Hz)": round(avg_pitch, 2),
        "Silence (%)": round(silence_pct, 1),
        "Clipping (%)": round(clipping_pct, 2),
        "Low Freq Energy": round(low_energy, 3),
        "Mid Freq Energy": round(mid_energy, 3),
        "High Freq Energy": round(high_energy, 3),
        "Waveform Path": waveform_path
    }
