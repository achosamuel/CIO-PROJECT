"""
Streamlit frontend for:
- Upload OR record microphone audio (sounddevice)
- Send audio to FastAPI backend (/collective or /analyze)
- Display scores, idea map, transcript, timeline
- NEW: bar chart per speaker (speaking minutes + interruption minutes)

Run:
  python -m streamlit run frontend_app.py
"""

from __future__ import annotations

import io
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import requests
import sounddevice as sd
import streamlit as st

from utils_audio import save_numpy_audio_to_wav


# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(page_title="Collective Organization Analyzer", layout="wide")
st.title("Collective Organization Analyzer (Offline + Free)")


# -------------------------
# Helpers
# -------------------------
def safe_post(url: str, files: dict, data: dict, timeout: int = 600) -> Dict[str, Any]:
    """POST with clean error messages and stable connection handling."""
    try:
        r = requests.post(
            url,
            files=files,
            data=data,
            timeout=(10, timeout),  # (connect, read)
            headers={"Connection": "close"},
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach backend: {exc}") from exc

    if r.status_code >= 400:
        try:
            payload = r.json()
            detail = payload.get("detail", payload)
        except Exception:
            detail = r.text
        raise RuntimeError(f"Backend error ({r.status_code}): {detail}")
    return r.json()


def seconds_to_mmss(x: float) -> str:
    m = int(x // 60)
    s = int(x % 60)
    return f"{m:02d}:{s:02d}"


def show_json_collapsible(label: str, obj: Any):
    with st.expander(label, expanded=False):
        st.code(json.dumps(obj, indent=2, ensure_ascii=False), language="json")


def record_microphone_wav(seconds: int, sample_rate: int = 16000) -> bytes:
    """Record from default microphone using sounddevice and return WAV bytes."""
    if seconds <= 0:
        raise ValueError("seconds must be > 0")

    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    audio = np.squeeze(audio)  # (N,)
    if audio.size == 0:
        raise RuntimeError("No audio captured from microphone.")

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "recording.wav"
        save_numpy_audio_to_wav(audio, wav_path, sample_rate=sample_rate)
        return wav_path.read_bytes()


def compute_speaking_and_interruptions(
    segments: list[dict[str, Any]],
    interruptions: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Compute:
    - Speaking time per speaker from diarization segments
    - Interruption time per speaker credited to the *interrupter* using overlap_duration
    Output in minutes.
    """
    speak: dict[str, float] = {}
    for s in segments or []:
        spk = str(s.get("speaker", "UNKNOWN"))
        start = float(s.get("start", 0.0) or 0.0)
        end = float(s.get("end", 0.0) or 0.0)
        dur = max(0.0, end - start)
        speak[spk] = speak.get(spk, 0.0) + dur

    intr: dict[str, float] = {}
    for it in interruptions or []:
        spk = str(it.get("interrupter", "UNKNOWN"))
        dur = float(it.get("overlap_duration", 0.0) or 0.0)
        if dur > 0:
            intr[spk] = intr.get(spk, 0.0) + dur

    speakers = sorted(set(list(speak.keys()) + list(intr.keys())))
    rows = []
    for spk in speakers:
        rows.append(
            {
                "speaker": spk,
                "speaking_min": speak.get(spk, 0.0) / 60.0,
                "interruption_min": intr.get(spk, 0.0) / 60.0,
            }
        )
    return pd.DataFrame(rows)


# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Settings")

backend_url = (
    st.sidebar.text_input(
        "Backend base URL",
        value="http://127.0.0.1:8000",
        help="Example: http://127.0.0.1:8000",
    )
    .strip()
    .rstrip("/")
)

endpoint = st.sidebar.selectbox(
    "Mode",
    ["Collective (ideas + scores)", "Analyze (diarization only)"],
    index=0,
)

record_seconds = st.sidebar.slider("Record duration (seconds)", 5, 120, 30, 5)

min_overlap = st.sidebar.slider(
    "Min overlap threshold (seconds)",
    0.0,
    2.0,
    0.2,
    0.05,
    help="Overlap must be at least this long to count as an overlap/interrupt event.",
)

cut_in_window = st.sidebar.slider(
    "Cut-in window (seconds)",
    0.1,
    3.0,
    1.0,
    0.1,
    help="If a new speaker starts within this window at an overlap start, they are considered the interrupter.",
)

st.sidebar.divider()

# Health check
health_ok = False
try:
    h = requests.get(f"{backend_url}/health", timeout=3)
    health_ok = h.status_code == 200
except Exception:
    health_ok = False

st.sidebar.write("Backend status:", "✅ Online" if health_ok else "❌ Offline")
st.sidebar.caption("If Offline: start backend with uvicorn and check URL/port.")


# -------------------------
# Input audio (Upload OR Record)
# -------------------------
st.subheader("1) Provide audio")

mode = st.radio(
    "Choose input method",
    ["Upload audio", "Record from PC microphone"],
    horizontal=True,
)

audio_bytes: bytes | None = None
audio_name: str | None = None
audio_mime: str = "application/octet-stream"

if mode == "Upload audio":
    uploaded_file = st.file_uploader(
        "Upload an audio file (wav/mp3/m4a/flac/ogg)",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
    )

    if uploaded_file is not None:
        audio_bytes = uploaded_file.getvalue()
        audio_name = uploaded_file.name
        st.audio(audio_bytes)
        st.caption(f"File: {audio_name} • Size: {len(audio_bytes)/1024:.1f} KB")

else:
    st.info(
        "Records from your **computer microphone** using Python (sounddevice).\n"
        "Click **Record now**, speak for the selected duration, then click **Run**."
    )

    if "mic_audio" not in st.session_state:
        st.session_state.mic_audio = None

    if st.button("🎙️ Record now", type="primary"):
        try:
            with st.spinner(f"Recording for {record_seconds}s..."):
                wav_bytes = record_microphone_wav(record_seconds, sample_rate=16000)
            st.session_state.mic_audio = wav_bytes
            st.success("Recording complete.")
        except Exception as exc:
            st.session_state.mic_audio = None
            st.error(f"Recording failed: {exc}")

    if st.session_state.mic_audio:
        audio_bytes = st.session_state.mic_audio
        audio_name = "recording.wav"
        audio_mime = "audio/wav"
        st.audio(audio_bytes, format="audio/wav")
        st.caption(f"Recorded: {len(audio_bytes)/1024:.1f} KB")

if not audio_bytes or not audio_name:
    st.stop()


# -------------------------
# Analyze button (lock to avoid Streamlit reruns)
# -------------------------
st.subheader("2) Run analysis")

if "busy" not in st.session_state:
    st.session_state.busy = False
if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None
if "elapsed" not in st.session_state:
    st.session_state.elapsed = None

run = st.button("Run", type="primary", use_container_width=True, disabled=st.session_state.busy)

if run:
    st.session_state.busy = True
    st.session_state.result = None
    st.session_state.error = None
    st.session_state.elapsed = None

api_path = "/collective" if endpoint.startswith("Collective") else "/analyze"
api_url = f"{backend_url}{api_path}"

if st.session_state.busy:
    with st.spinner("Analyzing... (first run can be slower)"):
        t0 = time.time()
        try:
            result = safe_post(
                api_url,
                files={"file": (audio_name, io.BytesIO(audio_bytes), audio_mime)},
                data={
                    "min_overlap_threshold": str(min_overlap),
                    "cut_in_window": str(cut_in_window),
                },
                timeout=900,
            )
            st.session_state.result = result
        except Exception as exc:
            st.session_state.error = str(exc)
        finally:
            st.session_state.busy = False
            st.session_state.elapsed = time.time() - t0

if st.session_state.error:
    st.error(st.session_state.error)
    st.stop()

result = st.session_state.result
if not result:
    st.stop()

st.success(f"Analysis complete in {st.session_state.elapsed:.1f}s")


# -------------------------
# Common: Speakers summary
# -------------------------
st.subheader("3) Results")

num_speakers = int(result.get("num_speakers", 0) or 0)
speakers = result.get("speakers", []) or []
st.write(f"**Speakers detected:** {num_speakers}")
if speakers:
    st.caption(", ".join(speakers))

segments = result.get("segments", []) or []
overlaps = result.get("overlaps", []) or []
interruptions = result.get("interruptions", []) or []


# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["Scores", "Idea Map", "Transcript", "Timeline", "Charts", "Raw JSON"])

# ---- Scores tab
with tabs[0]:
    if "scores" not in result:
        st.info("Scores are only available in **Collective** mode.")
    else:
        scores = result["scores"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall", scores.get("collective_organization", 0))
        c2.metric("Independence", scores.get("independence", 0))
        c3.metric("Participation Balance", scores.get("participation_balance", 0))
        c4.metric("Idea Diversity", scores.get("idea_diversity", 0))

        debug = scores.get("debug", {}) or {}
        if debug.get("analysis_error"):
            st.warning(f"Analysis warning: {debug['analysis_error']}")

        if debug:
            df = pd.DataFrame([{"metric": k, "value": v} for k, v in debug.items()])
            st.markdown("**Debug metrics**")
            st.dataframe(df, use_container_width=True)

# ---- Idea map tab
with tabs[1]:
    if "idea_map" not in result:
        st.info("Idea map is only available in **Collective** mode.")
    else:
        idea_map = result.get("idea_map", {}) or {}
        main_ideas = idea_map.get("main_ideas", []) or []

        if not main_ideas:
            st.warning("No ideas extracted. Try a longer brainstorming audio (30–90s).")
        else:
            st.markdown("### Main ideas")
            for idea in main_ideas:
                title = idea.get("title", "Untitled")
                summary = idea.get("summary", "")
                with st.expander(f"🧠 {title}", expanded=False):
                    if summary:
                        st.write(summary)

                    sub_ideas = idea.get("sub_ideas", []) or []
                    if not sub_ideas:
                        st.caption("No sub-ideas.")
                    else:
                        rows = []
                        for s in sub_ideas:
                            rows.append(
                                {
                                    "sub_idea": s.get("text", ""),
                                    "speakers": ", ".join(s.get("speakers", []) or []),
                                    "evidence": (s.get("evidence", [""]) or [""])[0],
                                }
                            )
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ---- Transcript tab
with tabs[2]:
    if "transcript" not in result:
        st.info("Transcript is only available in **Collective** mode.")
    else:
        transcript = result.get("transcript", {}) or {}
        full_text = transcript.get("full_text", "") or ""
        segs = transcript.get("segments", []) or []

        if not full_text and not segs:
            st.warning("Transcript is empty. Check faster-whisper installation.")
        else:
            st.markdown("### Full transcript (preview)")
            st.text_area("", value=full_text[:8000], height=220)

            st.markdown("### Transcript segments")
            if segs:
                df = pd.DataFrame(segs)
                st.dataframe(df, use_container_width=True)
            else:
                st.caption("No timestamped segments.")

# ---- Timeline tab
with tabs[3]:
    if not segments:
        st.warning("No diarization segments available.")
    else:
        st.markdown("### Speaker timeline (table)")
        df = pd.DataFrame(segments)
        df["start_mmss"] = df["start"].apply(seconds_to_mmss)
        df["end_mmss"] = df["end"].apply(seconds_to_mmss)
        st.dataframe(df[["speaker", "start", "end", "start_mmss", "end_mmss"]], use_container_width=True)

        st.markdown("### Overlaps")
        if overlaps:
            st.dataframe(pd.DataFrame(overlaps), use_container_width=True)
        else:
            st.caption("No overlaps detected with current thresholds.")

        st.markdown("### Interruptions")
        if interruptions:
            st.dataframe(pd.DataFrame(interruptions), use_container_width=True)
        else:
            st.caption("No interruptions detected with current thresholds.")

# ---- Charts tab
with tabs[4]:
    if not segments:
        st.warning("No diarization segments available for charts.")
    else:
        df_chart = compute_speaking_and_interruptions(segments, interruptions)

        if df_chart.empty:
            st.warning("Not enough data to draw charts.")
        else:
            st.markdown("### Speaking vs interruption time (minutes)")

            import matplotlib.pyplot as plt

            df_chart = df_chart.sort_values("speaker").reset_index(drop=True)
            speakers_list = df_chart["speaker"].tolist()
            x = np.arange(len(speakers_list))
            width = 0.38

            fig, ax = plt.subplots(figsize=(10, 4))

            # One unique color per speaker
            cmap = plt.get_cmap("tab10")
            colors = [cmap(i % 10) for i in range(len(speakers_list))]

            ax.bar(
                x - width / 2,
                df_chart["speaking_min"].values,
                width,
                label="Speaking (min)",
                color=colors,
            )
            ax.bar(
                x + width / 2,
                df_chart["interruption_min"].values,
                width,
                label="Interrupting (min)",
                color=colors,
                alpha=0.45,
            )

            ax.set_xticks(x)
            ax.set_xticklabels(speakers_list, rotation=30, ha="right")
            ax.set_ylabel("Minutes")
            ax.set_title("Per-speaker speaking time and interruption time")
            ax.legend()

            st.pyplot(fig, clear_figure=True)
            st.dataframe(df_chart, use_container_width=True)

# ---- Raw JSON tab
with tabs[5]:
    show_json_collapsible("Full response JSON", result)
