import streamlit as st
import tempfile
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from diagnose_audio import diagnose_audio

st.set_page_config(page_title="Audio Diagnostic Tool", page_icon="🎧", layout="centered")

# Header and branding
st.image("logo.png", width=150)
st.title("🔊 Audio Diagnostic Tool")
st.caption("Upload an audio file to analyze its duration, volume, pitch, silence, clipping, and EQ balance.")

st.markdown("---")

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

            # Format duration to mm:ss
            total_seconds = int(result["Duration (sec)"])
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            duration_str = f"{minutes}:{seconds:02d}"

            # Update results with formatted values
            display_data = {
                "Duration": duration_str,
                "Sample Rate (Hz)": result["Sample Rate (Hz)"],
                "Bit Depth": result["Bit Depth"],
                "Average Volume (RMS)": round(result["Average Volume (RMS)"] * 1000),  # scaled for clarity
                "Average Pitch (Hz)": int(round(result["Average Pitch (Hz)"])) if not np.isnan(result["Average Pitch (Hz)"]) else "N/A",
                "Silence (%)": int(round(result["Silence (%)"])),
                "Clipping (%)": int(round(result["Clipping (%)"]))
            }

            def highlight(val, label):
                if label == "Average Volume (RMS)":
                    if val < 10:
                        return 'background-color: #fde2e2'
                    elif val > 300:
                        return 'background-color: #fff4cc'
                    else:
                        return 'background-color: #e2f7e2'
                if label == "Clipping (%)" and isinstance(val, (int, float)) and val > 0:
                    return 'background-color: #ffcccc'
                return ''

            df = pd.DataFrame(display_data.items(), columns=["Metric", "Value"])
            styled_df = df.style.apply(
                lambda row: [highlight(row["Value"], row["Metric"]), ""],
                axis=1
            )
            st.dataframe(styled_df, use_container_width=True)

            # EQ Balance - Horizontal Bar Chart
            st.subheader("🎚️ Frequency Balance (EQ Snapshot)")
            labels = ["Low", "Mid", "High"]
            values = [
                result["Low Freq Energy"],
                result["Mid Freq Energy"],
                result["High Freq Energy"]
            ]
            fig, ax = plt.subplots(figsize=(6, 3))
            bars = ax.barh(labels, values, color=["#7da6ff", "#72d572", "#ffa726"])
            ax.set_xlabel("Energy")
            ax.set_xlim(left=0)
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{width:.1f}", va='center')
            ax.set_title("Frequency Balance")
            st.pyplot(fig)

            # Static waveform snapshot
            if os.path.exists(result["Waveform Path"]):
                st.subheader("🖼️ Waveform Snapshot")
                st.image(result["Waveform Path"], use_column_width=True)

            # Spectrogram
            if os.path.exists(result["Spectrogram Path"]):
                st.subheader("🌈 Spectrogram (Log Scale)")
                st.image(result["Spectrogram Path"], use_column_width=True)

        except Exception as e:
            st.error(f"❌ Error analyzing audio: {e}")
        finally:
            os.remove(tmp_path)
