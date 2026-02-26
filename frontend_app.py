"""Streamlit frontend for diarization + collective organization analysis.

Local run:
- Set backend env vars first (PowerShell): $env:OPENAI_API_KEY="..." ; $env:HF_TOKEN="..."
- Backend: uvicorn backend_app:app --host 127.0.0.1 --port 8000
- Frontend: streamlit run frontend_app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import sounddevice as sd
import streamlit as st

from utils_audio import save_numpy_audio_to_wav

st.set_page_config(page_title="Brainstorming Facilitator", layout="wide")
st.title("🎙️ Brainstorming Facilitator")

backend_url = st.sidebar.text_input("Backend URL", value="http://localhost:8000/collective")
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


if st.button("Analyze + Collective Organization"):
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
                    st.write(f"**Number of speakers:** {result.get('num_speakers', 0)}")
                    st.write(
                        f"**Speakers:** {', '.join(result.get('speakers', [])) if result.get('speakers') else 'None'}"
                    )

                    st.subheader("Per-speaker speaking time")
                    totals = compute_speaker_totals(result.get("segments", []))
                    totals_df = pd.DataFrame(
                        [{"speaker": s, "total_seconds": round(t, 3)} for s, t in totals.items()]
                    )
                    if not totals_df.empty:
                        totals_df = totals_df.sort_values("total_seconds", ascending=False)
                    st.dataframe(totals_df, use_container_width=True)

                    st.subheader("Collective Organization Score")
                    scores = result.get("scores", {})
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Overall", scores.get("collective_organization", "-"))
                    s2.metric("Independence", scores.get("independence", "-"))
                    s3.metric("Participation Balance", scores.get("participation_balance", "-"))
                    s4.metric("Idea Diversity", scores.get("idea_diversity", "-"))

                    debug_metrics = scores.get("debug", {})
                    if debug_metrics:
                        st.caption("Debug metrics")
                        debug_df = pd.DataFrame(
                            [{"metric": key, "value": value} for key, value in debug_metrics.items()]
                        )
                        st.dataframe(debug_df, use_container_width=True)

                    st.subheader("Idea Map")
                    main_ideas = result.get("idea_map", {}).get("main_ideas", [])
                    if not main_ideas:
                        st.write("No ideas extracted.")
                    for main_idea in main_ideas:
                        title = main_idea.get("title", "Untitled")
                        st.markdown(f"**{main_idea.get('id', 'I?')} — {title}**")
                        st.write(main_idea.get("summary", ""))
                        with st.expander("Sub-ideas", expanded=False):
                            for idx, sub_idea in enumerate(main_idea.get("sub_ideas", []), start=1):
                                st.markdown(f"{idx}. {sub_idea.get('text', '')}")
                                st.caption(f"Speakers: {', '.join(sub_idea.get('speakers', [])) or 'UNKNOWN'}")
                                evidence = sub_idea.get("evidence", [])
                                if evidence:
                                    st.markdown("Evidence:")
                                    for quote in evidence:
                                        st.markdown(f"- \"{quote}\"")

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
    "Backend needs HF_TOKEN and OPENAI_API_KEY. OpenAI calls are server-side only in backend_app.py."
)
