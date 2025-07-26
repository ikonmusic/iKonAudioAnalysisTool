st.image("logo.png", width=150)
st.title("🔎 Audio Diagnostic Tool")
st.caption("Built to help you inspect audio quality in seconds.")
import streamlit as st
import tempfile
import os
from diagnose_audio import diagnose_audio

st.set_page_config(page_title="Audio Diagnostic Tool")

st.title("🔎 Audio Diagnostic Tool")
st.write("Upload a WAV file and get audio analysis.")

uploaded_file = st.file_uploader("Upload your audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file, format="audio/wav")
    
    try:
        result = diagnose_audio(tmp_path)
        st.subheader("📊 Results")
        st.json(result)
    except Exception as e:
        st.error(f"Error analyzing audio: {e}")
    finally:
        os.remove(tmp_path)
