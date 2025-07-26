import streamlit as st
import tempfile
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from diagnose_audio import diagnose_audio

# Streamlit page config
st.set_page_config(page_title="Audio Diagnostic Tool", page_icon="🎧", layout="centered")

# Header and branding
st.image("logo.png", width=150)
st.title("🔊 Audio Diagnostic Tool")
st.caption("Upload an audio file to analyze its duration, volume, pitch, silence, and frequency balance.")

st.markdown("---")

# File uploader
uploaded_file = st.file_uploader("🎵 Upload your audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("🔎 Analyzing audio..."):
        try:
            result = diagnose_audio(tmp_path)
            st.success("✅ Analysis complete!")
            st.subheader("📊 Diagnostic Results")

            # Color-coded table
            def highlight(val, label):
                if label == "Average Volume (RMS)":
                    if val < 0.01:
                        return 'background-color: #fde2e2'  # too quiet
                    elif val > 0.3:
                        return 'background-color: #fff4cc'  # maybe clipping
                    else:
                        return 'background-color: #e2f7e2'  # good
                if label == "Average Pitch (Hz)":
                    if np.isnan(val) or val < 50:
                        return 'background-color: #fde2e2'  # pitch too low or missing
                return ''

            df = pd.DataFrame(result.items(), columns=["Metric", "Value"])
            styled_df = df.style.applymap(
                lambda val: highlight(val, df.loc[df["Value"] == val, "Metric"].values[0])
            )
            st.dataframe(styled_df, use_container_width=True)

            # Frequency balance plot
            st.subheader("🎚️ Frequency Balance (EQ Snapshot)")
            low = result["Low Freq Energy"]
            mid = result["Mid Freq Energy"]
            high = result["High Freq Energy"]

            fig, ax = plt.subplots()
            ax.bar(["Low", "Mid", "High"], [low, mid, high], color=["#7da6ff", "#72d572", "#ffa726"])
            ax.set_ylabel("Energy")
            ax.set_title("Frequency Balance")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error analyzing audio: {e}")
        finally:
            os.remove(tmp_path)
