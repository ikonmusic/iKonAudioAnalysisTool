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
st.caption("Upload an audio file to analyze duration, volume, pitch, silence, clipping, and EQ balance.")

st.markdown("---")

uploaded_file = st.file_uploader("🎵 Upload your audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("🔎 Analyzing audio..."):
        try:
            result = diagnose_audio(tmp_path)
            st.success("✅ Analysis complete!")
            st.subheader("📊 Diagnostic Results")

            # Format duration as mm:ss
            total_seconds = int(result["Duration (sec)"])
            duration_str = f"{total_seconds // 60}:{total_seconds % 60:02d}"

            # Display metrics
            display_data = {
                "Duration": duration_str,
                "Sample Rate (Hz)": result["Sample Rate (Hz)"],
                "Bit Depth": result["Bit Depth"],
                "Average Volume (RMS)": round(result["Average Volume (RMS)"] * 1000),
                "Average Pitch (Hz)": int(round(result["Average Pitch (Hz)"])) if not np.isnan(result["Average Pitch (Hz)"]) else "N/A",
                "Silence (%)": int(round(result["Silence (%)"])),
                "Clipping (%)": int(round(result["Clipping (%)"]))
            }

            def highlight(val, label):
                if label == "Average Volume (RMS)":
                    if val < 10:
                        return 'background-color: #fde2e2'
                    elif val > 300:
                        return 'background-color: #FF0000'
                    else:
                        return 'background-color: #e2f7e2'
                if label == "Clipping (%)" and isinstance(val, (int, float)) and val > 0:
                    return 'background-color: #FF0000'
                return ''

            df = pd.DataFrame(display_data.items(), columns=["Metric", "Value"])
            styled_df = df.style.apply(
                lambda row: [highlight(row["Value"], row["Metric"]), ""],
                axis=1
            )
            st.dataframe(styled_df, use_container_width=True)

            # EQ Balance - Frequency Curve Plot (No energy)
            st.subheader("🎚️ Frequency Bands (Visual Reference)")

            freqs = np.linspace(20, 16000, 1000)

            def bell_mask(x, mu, sigma):
                return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

            low_curve = bell_mask(freqs, mu=100, sigma=80)
            mid_curve = bell_mask(freqs, mu=1000, sigma=700)
            high_curve = bell_mask(freqs, mu=8000, sigma=3000)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(freqs, low_curve, label="Low Band", color="#7da6ff", linewidth=2)
            ax.plot(freqs, mid_curve, label="Mid Band", color="#72d572", linewidth=2)
            ax.plot(freqs, high_curve, label="High Band", color="#ffa726", linewidth=2)

            ax.set_xscale('log')
            ax.set_xlim(20, 16000)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Relative Response")
            ax.set_title("Frequency Bands Curve")
            ax.legend()
            ax.grid(True, which="both", linestyle="--", linewidth=0.3)
            st.pyplot(fig)

            # Waveform preview
            if os.path.exists(result["Waveform Path"]):
                st.subheader("🗆️ Waveform Snapshot")
                st.image(result["Waveform Path"], use_column_width=True)

            # Spectrogram preview
            if os.path.exists(result["Spectrogram Path"]):
                st.subheader("🌈 Spectrogram (Log Scale)")
                st.image(result["Spectrogram Path"], use_column_width=True)

        except Exception as e:
            st.error(f"❌ Error analyzing audio: {e}")
        finally:
            os.remove(tmp_path)
