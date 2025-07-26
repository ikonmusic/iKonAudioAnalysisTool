import streamlit as st
import tempfile
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
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

            # Color-coded metric table
            def highlight(val, label):
                if label == "Average Volume (RMS)":
                    if val < 0.01:
                        return 'background-color: #fde2e2'
                    elif val > 0.3:
                        return 'background-color: #fff4cc'
                    else:
                        return 'background-color: #e2f7e2'
                if label == "Clipping (%)" and val > 0.5:
                    return 'background-color: #ffcccc'
                if label == "Silence (%)" and val > 70:
                    return 'background-color: #fff4cc'
                if label == "Average Pitch (Hz)" and (np.isnan(val) or val < 50):
                    return 'background-color: #fde2e2'
                return ''

            # Show diagnostic values
            display_keys = [
                "Duration (sec)", "Sample Rate (Hz)", "Bit Depth", "Average Volume (RMS)",
                "Average Pitch (Hz)", "Silence (%)", "Clipping (%)"
            ]
            df = pd.DataFrame([[k, result[k]] for k in display_keys], columns=["Metric", "Value"])
            styled_df = df.style.apply(
                lambda row: [highlight(row["Value"], row["Metric"]), ""],
                axis=1
            )
            st.dataframe(styled_df, use_container_width=True)

            # Frequency balance bar chart
            st.subheader("🎚️ Frequency Balance (EQ Snapshot)")
            fig_eq, ax_eq = plt.subplots()
            ax_eq.bar(["Low", "Mid", "High"], 
                      [result["Low Freq Energy"], result["Mid Freq Energy"], result["High Freq Energy"]],
                      color=["#7da6ff", "#72d572", "#ffa726"])
            ax_eq.set_ylabel("Energy")
            ax_eq.set_title("Frequency Balance")
            st.pyplot(fig_eq)

            # Waveform preview — Interactive
            st.subheader("📈 Interactive Waveform")
            wave_data = result.get("Waveform Data")
            if wave_data:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=wave_data["times"],
                    y=wave_data["amplitudes"],
                    mode="lines",
                    line=dict(color="RoyalBlue"),
                    name="Waveform"
                ))
                fig.update_layout(
                    xaxis_title="Time (s)",
                    yaxis_title="Amplitude",
                    height=300,
                    margin=dict(l=20, r=20, t=30, b=20),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

            # Static waveform (optional)
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
