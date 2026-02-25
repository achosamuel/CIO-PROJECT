"""Streamlit frontend for recording/uploading audio and showing diarization insights."""

from __future__ import annotations

import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import sounddevice as sd
import streamlit as st

from utils_audio import save_numpy_audio_to_wav

st.set_page_config(page_title="Speaker Counter + Interruption Detector", layout="wide")
st.title("🎙️ Speaker Counter + Interruption Detector")

backend_url = st.sidebar.text_input("Backend URL", value="http://localhost:8000/analyze")
record_duration = st.sidebar.slider("Record duration (seconds)", min_value=5, max_value=60, value=10)
min_overlap_threshold = st.sidebar.slider("Min overlap threshold (seconds)", 0.0, 2.0, 0.2, 0.1)
cut_in_window = st.sidebar.slider("Cut-in window (seconds)", 0.1, 3.0, 1.0, 0.1)
sample_rate = 16_000

if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

col1, col2 = st.columns(2)

with col1:
    st.subheader("Record from microphone")
    if st.button("Record"):
        with st.spinner(f"Recording {record_duration} seconds..."):
            recording = sd.rec(
                int(record_duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()

            temp_path = Path(tempfile.gettempdir()) / "streamlit_recording.wav"
            save_numpy_audio_to_wav(recording.squeeze(), temp_path, sample_rate=sample_rate)
            st.session_state.audio_path = str(temp_path)
        st.success(f"Recording saved to {st.session_state.audio_path}")

with col2:
    st.subheader("Upload audio file")
    uploaded_file = st.file_uploader("Choose WAV/MP3/M4A audio", type=["wav", "mp3", "m4a", "flac", "ogg"])
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix or ".wav"
        temp_upload_path = Path(tempfile.gettempdir()) / f"uploaded_audio{suffix}"
        temp_upload_path.write_bytes(uploaded_file.read())
        st.session_state.audio_path = str(temp_upload_path)
        st.success(f"Uploaded file ready: {uploaded_file.name}")

selected = st.session_state.audio_path
if selected:
    st.info(f"Selected audio: {selected}")


def compute_speaker_totals(segments: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for seg in segments:
        duration = float(seg["end"]) - float(seg["start"])
        totals[seg["speaker"]] = totals.get(seg["speaker"], 0.0) + max(duration, 0.0)
    return totals


def draw_timeline(segments: list[dict]) -> None:
    if not segments:
        st.write("No timeline to display.")
        return

    speakers = sorted({seg["speaker"] for seg in segments})
    speaker_to_y = {spk: idx for idx, spk in enumerate(speakers)}

    fig, ax = plt.subplots(figsize=(10, max(3, len(speakers))))

    cmap = plt.get_cmap("tab10")
    speaker_to_color = {spk: cmap(i % 10) for i, spk in enumerate(speakers)}

    for seg in segments:
        y = speaker_to_y[seg["speaker"]]
        start = float(seg["start"])
        width = float(seg["end"] - seg["start"])
        ax.broken_barh([(start, width)], (y - 0.4, 0.8), facecolors=speaker_to_color[seg["speaker"]])

    ax.set_xlabel("Time (seconds)")
    ax.set_yticks(list(speaker_to_y.values()))
    ax.set_yticklabels(list(speaker_to_y.keys()))
    ax.set_title("Speaker timeline")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    st.pyplot(fig)


if st.button("Analyze"):
    if not selected:
        st.error("Please record or upload an audio file first.")
    else:
        with st.spinner("Sending audio to backend for analysis..."):
            try:
                with open(selected, "rb") as f:
                    response = requests.post(
                        backend_url,
                        files={"file": (Path(selected).name, f, "audio/wav")},
                        data={
                            "min_overlap_threshold": min_overlap_threshold,
                            "cut_in_window": cut_in_window,
                        },
                        timeout=600,
                    )

                if response.status_code != 200:
                    st.error(f"Backend error ({response.status_code}): {response.text}")
                else:
                    result = response.json()
                    st.success("Analysis complete.")

                    st.subheader("Summary")
                    st.write(f"**Number of speakers:** {result['num_speakers']}")
                    st.write(f"**Speakers:** {', '.join(result['speakers']) if result['speakers'] else 'None'}")

                    st.subheader("Per-speaker speaking time")
                    totals = compute_speaker_totals(result.get("segments", []))
                    totals_df = pd.DataFrame(
                        [{"speaker": s, "total_seconds": round(t, 3)} for s, t in totals.items()]
                    ).sort_values("total_seconds", ascending=False)
                    st.dataframe(totals_df, use_container_width=True)

                    st.subheader("Interruptions")
                    interruptions = result.get("interruptions", [])
                    if interruptions:
                        st.dataframe(pd.DataFrame(interruptions), use_container_width=True)
                    else:
                        st.write("No interruptions detected with the current thresholds.")

                    st.subheader("Timeline")
                    draw_timeline(result.get("segments", []))

            except requests.RequestException as exc:
                st.error(f"Could not reach backend. Check URL and server status. Error: {exc}")
            except Exception as exc:
                st.error(f"Unexpected frontend error: {exc}")

st.caption(
    "If diarization fails, make sure backend has HF_TOKEN set and that your account has access "
    "to pyannote models."
)
