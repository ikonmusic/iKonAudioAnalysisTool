# 🎈 Blank app template

A simple Streamlit app template for you to modify!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
pip install streamlit librosa soundfile numpy
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
