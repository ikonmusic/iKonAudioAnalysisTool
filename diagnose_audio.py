import librosa
import numpy as np

def diagnose_audio(file_path):
    # Load audio
    y, sr = librosa.load(file_path)

    # Duration
    duration = librosa.get_duration(y=y, sr=sr)

    # Volume (RMS)
    rms = np.mean(librosa.feature.rms(y=y))

    # Pitch detection using YIN algorithm
    try:
        pitch = librosa.yin(y, fmin=50, fmax=300)
        avg_pitch = np.mean(pitch)
    except:
        avg_pitch = float("nan")

    # Silence % (frames below a threshold RMS)
    rms_vals = librosa.feature.rms(y=y)[0]
    silence_thresh = 0.02
    silence_frames = np.sum(rms_vals < silence_thresh)
    silence_pct = (silence_frames / len(rms_vals)) * 100

    # Frequency spectrum
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    # Split energy into low, mid, and high frequency bands
    low_band = stft[(freqs >= 20) & (freqs < 250)]
    mid_band = stft[(freqs >= 250) & (freqs < 4000)]
    high_band = stft[(freqs >= 4000) & (freqs < 16000)]

    low_energy = np.mean(low_band) if low_band.size else 0
    mid_energy = np.mean(mid_band) if mid_band.size else 0
    high_energy = np.mean(high_band) if high_band.size else 0

    # Round and return as dictionary
    return {
        "Duration (sec)": round(duration, 2),
        "Average Volume (RMS)": round(rms, 6),
        "Average Pitch (Hz)": round(avg_pitch, 2),
        "Silence (%)": round(silence_pct, 1),
        "Low Freq Energy": round(low_energy, 3),
        "Mid Freq Energy": round(mid_energy, 3),
        "High Freq Energy": round(high_energy, 3)
    }
