import librosa
import numpy as np

def diagnose_audio(file_path):
    y, sr = librosa.load(file_path)
    duration = librosa.get_duration(y=y, sr=sr)
    rms = np.mean(librosa.feature.rms(y=y))
    pitch = librosa.yin(y, fmin=50, fmax=300)
    avg_pitch = np.mean(pitch)

    silence_thresh = 0.02
    rms_vals = librosa.feature.rms(y=y)[0]
    silence_frames = np.sum(rms_vals < silence_thresh)
    silence_pct = (silence_frames / len(rms_vals)) * 100

    return {
        "Duration (sec)": round(duration, 2),
        "Average Volume (RMS)": round(rms, 6),
        "Average Pitch (Hz)": round(avg_pitch, 2),
        "Silence (%)": round(silence_pct, 1)
    }
