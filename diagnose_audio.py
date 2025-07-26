import librosa
import numpy as np

def diagnose_audio(file_path):
    # Load audio
    y, sr = librosa.load(file_path, sr=None)

    if y.size == 0:
        raise ValueError("Audio file is empty or unreadable.")

    # Duration
    duration = librosa.get_duration(y=y, sr=sr)

    # Volume (RMS)
    rms_vals = librosa.feature.rms(y=y)
    if rms_vals.shape[1] == 0:
        raise ValueError("RMS calculation failed — possibly silent or empty audio.")

    rms = np.mean(rms_vals)

    # Pitch detection using YIN
    try:
        pitch = librosa.yin(y, fmin=50, fmax=300)
        avg_pitch = np.mean(pitch)
    except:
        avg_pitch = float("nan")

    # Silence %
    silence_thresh = 0.02
    silence_frames = np.sum(rms_vals[0] < silence_thresh)
    silence_pct = (silence_frames / len(rms_vals[0])) * 100

    # Frequency bands
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    def band_energy(freq_low, freq_high):
        band = stft[(freqs >= freq_low) & (freqs < freq_high)]
        return np.mean(band) if band.size > 0 else 0

    low_energy = band_energy(20, 250)
    mid_energy = band_energy(250, 4000)
    high_energy = band_energy(4000, 16000)

    return {
        "Duration (sec)": round(duration, 2),
        "Average Volume (RMS)": round(rms, 6),
        "Average Pitch (Hz)": round(avg_pitch, 2),
        "Silence (%)": round(silence_pct, 1),
        "Low Freq Energy": round(low_energy, 3),
        "Mid Freq Energy": round(mid_energy, 3),
        "High Freq Energy": round(high_energy, 3)
    }
