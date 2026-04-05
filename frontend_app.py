"""
Streamlit frontend for:
- Upload OR record microphone audio (sounddevice)
- Send audio to FastAPI backend (/collective or /analyze)
- Display scores, idea map, transcript, timeline
- NEW: polished UI, summary cards, guided workflow, and improved charts

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import sounddevice as sd
import streamlit as st
import streamlit.components.v1 as components

from utils_audio import save_numpy_audio_to_wav


# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(
    page_title="Collective Organization Analyzer",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------
# Design system
# -------------------------
CUSTOM_CSS = """
<style>
:root {
    --bg: #081120;
    --panel: rgba(12, 20, 38, 0.78);
    --panel-strong: rgba(17, 26, 48, 0.95);
    --panel-soft: rgba(255, 255, 255, 0.04);
    --text: #edf2ff;
    --muted: #aab6d3;
    --accent: #7c3aed;
    --accent-2: #14b8a6;
    --accent-3: #f59e0b;
    --border: rgba(255, 255, 255, 0.08);
    --shadow: 0 18px 50px rgba(2, 8, 23, 0.35);
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(124, 58, 237, 0.22), transparent 28%),
        radial-gradient(circle at top right, rgba(20, 184, 166, 0.18), transparent 24%),
        linear-gradient(180deg, #081120 0%, #0d1630 55%, #111827 100%);
    color: var(--text);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

h1, h2, h3, h4, h5, h6, p, span, label, div {
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10, 18, 35, 0.97), rgba(14, 24, 45, 0.95));
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.025));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 0.9rem 1rem;
    box-shadow: var(--shadow);
}

.stButton > button,
[data-testid="baseButton-primary"] {
    border-radius: 14px;
    border: none;
    background: linear-gradient(135deg, var(--accent), #2563eb);
    color: white !important;
    font-weight: 700;
    min-height: 3rem;
    box-shadow: 0 14px 30px rgba(37, 99, 235, 0.28);
}

.stDownloadButton > button,
.stButton > button:hover,
[data-testid="baseButton-primary"]:hover {
    filter: brightness(1.05);
    transform: translateY(-1px);
}

.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextArea"] textarea:disabled,
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"],
.stFileUploader,
.stSlider,
.stRadio,
.stExpander,
.stTabs [data-baseweb="tab-list"],
.stDataFrame,
[data-testid="stAlert"] {
    border-radius: 16px !important;
}

.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextArea"] textarea:disabled {
    background: rgba(12, 20, 38, 0.92) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    caret-color: var(--text) !important;
}

div[data-testid="stTextArea"] textarea::placeholder {
    color: var(--muted) !important;
}

[data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: rgba(255,255,255,0.035);
    padding: 0.4rem;
    border-radius: 18px;
    border: 1px solid var(--border);
}

button[data-baseweb="tab"] {
    border-radius: 12px;
    padding: 0.6rem 1rem;
    color: var(--muted);
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(124,58,237,0.23), rgba(37,99,235,0.18));
    color: white;
}

.glass-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 1.35rem;
    box-shadow: var(--shadow);
    backdrop-filter: blur(14px);
}

.hero-card {
    position: relative;
    overflow: hidden;
    padding: 1.7rem;
    border-radius: 28px;
    background:
        linear-gradient(135deg, rgba(124,58,237,0.22), rgba(37,99,235,0.18)),
        linear-gradient(180deg, rgba(15,23,42,0.88), rgba(15,23,42,0.72));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: var(--shadow);
}

.hero-card::after {
    content: "";
    position: absolute;
    inset: auto -40px -60px auto;
    width: 180px;
    height: 180px;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(20,184,166,0.3), transparent 70%);
}

.hero-title {
    font-size: clamp(2rem, 4vw, 3.4rem);
    line-height: 1.05;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 0.75rem;
}

.hero-subtitle {
    color: var(--muted);
    font-size: 1.02rem;
    max-width: 56rem;
    margin-bottom: 1rem;
}

.badge-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}

.badge-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.5rem 0.8rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.08);
    color: #e8ecff;
    font-size: 0.92rem;
}

.section-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin: 1.25rem 0 0.9rem;
}

.section-heading h2 {
    margin: 0;
    font-size: 1.2rem;
}

.section-copy {
    color: var(--muted);
    margin-top: 0.35rem;
    margin-bottom: 0;
}

.feature-list {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.9rem;
    margin-top: 1rem;
}

.feature-item {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 0.95rem;
}

.feature-item strong {
    display: block;
    margin-bottom: 0.35rem;
}

.helper-note {
    color: var(--muted);
    font-size: 0.93rem;
}

.empty-state {
    text-align: center;
    padding: 1.6rem 1rem;
    border-radius: 18px;
    background: rgba(255,255,255,0.03);
    border: 1px dashed rgba(255,255,255,0.12);
}

@media (max-width: 900px) {
    .feature-list {
        grid-template-columns: 1fr;
    }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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
            timeout=(10, timeout),
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

    audio = np.squeeze(audio)
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
    """Compute speaking and interruption time per speaker in minutes."""
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



def render_hero() -> None:
    st.markdown(
        """
        <section class="hero-card">
            <div class="badge-chip" style="width:max-content; margin-bottom: 1rem;">✨ Redesigned experience</div>
            <div class="hero-title">Beautiful audio analysis dashboard for teams, workshops, and meetings.</div>
            <p class="hero-subtitle">
                Upload or record audio, run speaker analysis, and review clear insights for participation,
                interruptions, idea clustering, and transcripts in one polished workspace.
            </p>
            <div class="badge-row">
                <div class="badge-chip">🎙️ Upload or record audio</div>
                <div class="badge-chip">🧠 Collective intelligence scoring</div>
                <div class="badge-chip">📊 Speaker activity visualization</div>
                <div class="badge-chip">⚡ FastAPI + Streamlit workflow</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )



def render_section_heading(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-heading">
            <div>
                <h2>{title}</h2>
                <p class="section-copy">{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_overview_panel(health_ok: bool, backend_url: str, endpoint: str) -> None:
    health_label = "Online" if health_ok else "Offline"
    health_dot = "🟢" if health_ok else "🔴"
    st.markdown(
        f"""
        <div class="glass-card">
            <h3 style="margin-top:0;">Session overview</h3>
            <p class="helper-note">A cleaner workflow modeled after modern analytics dashboards.</p>
            <div class="feature-list">
                <div class="feature-item">
                    <strong>{health_dot} Backend status</strong>
                    <span>{health_label}</span>
                </div>
                <div class="feature-item">
                    <strong>🧭 Active mode</strong>
                    <span>{endpoint}</span>
                </div>
                <div class="feature-item">
                    <strong>🔗 API target</strong>
                    <span>{backend_url}</span>
                </div>
                <div class="feature-item">
                    <strong>✅ Design goals</strong>
                    <span>Clarity, hierarchy, spacing, accessibility</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_empty_state(icon: str, title: str, message: str) -> None:
    st.markdown(
        f"""
        <div class="empty-state">
            <div style="font-size: 2rem; margin-bottom: 0.4rem;">{icon}</div>
            <div style="font-size: 1.05rem; font-weight: 700; margin-bottom: 0.35rem;">{title}</div>
            <div class="helper-note">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_summary_kpis(num_speakers: int, overlaps: list[dict[str, Any]], interruptions: list[dict[str, Any]], elapsed: float | None) -> None:
    elapsed_value = f"{elapsed:.1f}s" if elapsed is not None else "--"
    html = f"""
    <div id="kpi-root">
      <style>
        #kpi-root * {{ box-sizing: border-box; }}
        #kpi-root .kpi-grid {{
            display:grid;
            grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
            gap:14px;
            margin: 6px 0 18px;
        }}
        #kpi-root .kpi-card {{
            background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.04));
            border:1px solid rgba(255,255,255,0.08);
            border-radius:20px;
            padding:18px;
            box-shadow: 0 16px 35px rgba(2,8,23,0.22);
        }}
        #kpi-root .kpi-label {{ color:#aab6d3; font-size:13px; margin-bottom:6px; }}
        #kpi-root .kpi-value {{ color:#f8fbff; font-weight:800; font-size:34px; letter-spacing:-0.03em; }}
        #kpi-root .kpi-foot {{ color:#c8d2ea; font-size:12px; margin-top:6px; }}
      </style>
      <div class="kpi-grid">
        <div class="kpi-card"><div class="kpi-label">Detected speakers</div><div class="kpi-value" data-target="{num_speakers}">0</div><div class="kpi-foot">Unique voices identified</div></div>
        <div class="kpi-card"><div class="kpi-label">Overlap events</div><div class="kpi-value" data-target="{len(overlaps)}">0</div><div class="kpi-foot">Moments with simultaneous speech</div></div>
        <div class="kpi-card"><div class="kpi-label">Interruptions</div><div class="kpi-value" data-target="{len(interruptions)}">0</div><div class="kpi-foot">Detected cut-in events</div></div>
        <div class="kpi-card"><div class="kpi-label">Processing time</div><div class="kpi-value" data-target-text="{elapsed_value}">{elapsed_value}</div><div class="kpi-foot">End-to-end backend request</div></div>
      </div>
      <script>
        const counters = document.querySelectorAll('#kpi-root .kpi-value[data-target]');
        counters.forEach((node) => {{
          const target = Number(node.dataset.target || 0);
          const duration = 700;
          const start = performance.now();
          function tick(now) {{
            const progress = Math.min((now - start) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            node.textContent = Math.round(target * eased).toLocaleString();
            if (progress < 1) requestAnimationFrame(tick);
          }}
          requestAnimationFrame(tick);
        }});
      </script>
    </div>
    """
    components.html(html, height=170)



def render_audio_hint() -> None:
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="margin-top:0;">Tips for the best result</h3>
            <div class="feature-list">
                <div class="feature-item">
                    <strong>Use clear audio</strong>
                    <span>Reduce background noise and keep speakers close to the microphone.</span>
                </div>
                <div class="feature-item">
                    <strong>Choose the right mode</strong>
                    <span>Use Collective mode for scores and idea mapping, Analyze mode for diarization only.</span>
                </div>
                <div class="feature-item">
                    <strong>Longer clips help</strong>
                    <span>Brainstorming clips around 30–90 seconds improve topic and idea extraction.</span>
                </div>
                <div class="feature-item">
                    <strong>Tune overlap sensitivity</strong>
                    <span>Adjust the thresholds in the sidebar to better capture interruptions.</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------
# Header + layout intro
# -------------------------
render_hero()
st.write("")


# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.markdown("## Control center")
st.sidebar.caption("Set the backend, analysis mode, and interruption sensitivity.")

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
    "Analysis mode",
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
    help="Overlap must be at least this long to count as an overlap or interrupt event.",
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

health_ok = False
try:
    h = requests.get(f"{backend_url}/health", timeout=3)
    health_ok = h.status_code == 200
except Exception:
    health_ok = False

st.sidebar.write("Backend status:", "✅ Online" if health_ok else "❌ Offline")
st.sidebar.caption("If offline, start the FastAPI backend and verify the host and port.")

left_top, right_top = st.columns([1.35, 0.95], gap="large")
with left_top:
    render_section_heading(
        "1. Provide audio",
        "Choose a source, preview the file, and prepare the run with a cleaner and more modern interface.",
    )
with right_top:
    render_overview_panel(health_ok, backend_url, endpoint)


# -------------------------
# Input audio
# -------------------------
mode = st.radio(
    "Choose input method",
    ["Upload audio", "Record from PC microphone"],
    horizontal=True,
    label_visibility="collapsed",
)

audio_bytes: bytes | None = None
audio_name: str | None = None
audio_mime: str = "application/octet-stream"

col_input, col_help = st.columns([1.35, 0.9], gap="large")

with col_input:
    if mode == "Upload audio":
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Upload audio")
        st.caption("Supported formats: wav, mp3, m4a, flac, ogg.")
        uploaded_file = st.file_uploader(
            "Upload an audio file",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            audio_bytes = uploaded_file.getvalue()
            audio_name = uploaded_file.name
            audio_mime = uploaded_file.type or audio_mime
            st.audio(audio_bytes)
            file_size_kb = len(audio_bytes) / 1024
            info_cols = st.columns(2)
            info_cols[0].markdown(f"**File name**  \\n{audio_name}")
            info_cols[1].markdown(f"**Size**  \\n{file_size_kb:.1f} KB")
        else:
            render_empty_state("📁", "No audio uploaded yet", "Add a file to unlock the analysis workflow.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Record from microphone")
        st.caption("Record directly from your default microphone using Python sounddevice.")

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
            st.caption(f"Recorded audio size: {len(audio_bytes) / 1024:.1f} KB")
        else:
            render_empty_state("🎧", "No recording captured yet", "Click Record now, speak naturally, then run the analysis.")
        st.markdown("</div>", unsafe_allow_html=True)

with col_help:
    render_audio_hint()

if not audio_bytes or not audio_name:
    st.info("Provide an audio file or complete a microphone recording to continue.")
    st.stop()


# -------------------------
# Analyze button
# -------------------------
render_section_heading(
    "2. Run analysis",
    "Launch the backend request and keep the output organized in clear result sections.",
)

if "busy" not in st.session_state:
    st.session_state.busy = False
if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None
if "elapsed" not in st.session_state:
    st.session_state.elapsed = None

run_col, meta_col = st.columns([0.9, 1.1], gap="large")
with run_col:
    run = st.button("Run analysis", type="primary", use_container_width=True, disabled=st.session_state.busy)
with meta_col:
    st.markdown(
        f"""
        <div class="glass-card">
            <strong>Selected configuration</strong>
            <div class="helper-note" style="margin-top:0.45rem;">
                Mode: {endpoint}<br>
                Min overlap: {min_overlap:.2f}s<br>
                Cut-in window: {cut_in_window:.1f}s
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if run:
    st.session_state.busy = True
    st.session_state.result = None
    st.session_state.error = None
    st.session_state.elapsed = None

api_path = "/collective" if endpoint.startswith("Collective") else "/analyze"
api_url = f"{backend_url}{api_path}"

if st.session_state.busy:
    with st.spinner("Analyzing... the first run can be slower while models load."):
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
    render_empty_state("🚀", "Ready when you are", "Run analysis to generate scores, transcripts, and speaker insights.")
    st.stop()

st.success(f"Analysis complete in {st.session_state.elapsed:.1f}s")


# -------------------------
# Results summary
# -------------------------
render_section_heading(
    "3. Results",
    "A cleaner result hierarchy with KPI cards, focused tabs, and readable detail views.",
)

num_speakers = int(result.get("num_speakers", 0) or 0)
speakers = result.get("speakers", []) or []
segments = result.get("segments", []) or []
overlaps = result.get("overlaps", []) or []
interruptions = result.get("interruptions", []) or []

render_summary_kpis(num_speakers, overlaps, interruptions, st.session_state.elapsed)

if speakers:
    speaker_chips = " ".join(f"<span class=\"badge-chip\">👤 {speaker}</span>" for speaker in speakers)
    st.markdown(f"<div class=\"badge-row\">{speaker_chips}</div>", unsafe_allow_html=True)


# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["Overview", "Scores", "Idea Map", "Transcript", "Timeline", "Charts", "Raw JSON"])

with tabs[0]:
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown("### Analysis snapshot")
        snapshot_rows = [
            {"metric": "Speakers detected", "value": num_speakers},
            {"metric": "Overlap events", "value": len(overlaps)},
            {"metric": "Interruptions", "value": len(interruptions)},
            {"metric": "Segments", "value": len(segments)},
            {"metric": "Mode", "value": endpoint},
            {"metric": "Backend URL", "value": backend_url},
        ]
        st.dataframe(pd.DataFrame(snapshot_rows), use_container_width=True, hide_index=True)
    with right:
        st.markdown("### Recommendation")
        if num_speakers <= 1:
            render_empty_state("🗣️", "Add more participants", "This clip appears to contain one speaker or very limited turn-taking.")
        elif len(interruptions) > len(overlaps) * 0.5 and overlaps:
            render_empty_state("⚠️", "High interruption density", "Consider increasing facilitation or revisiting overlap thresholds for validation.")
        else:
            render_empty_state("✅", "Balanced session", "The session appears readable and ready for deeper score and transcript review.")

with tabs[1]:
    if "scores" not in result:
        render_empty_state("📈", "Scores unavailable", "Collective mode is required to calculate organization scores.")
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
            debug_df = pd.DataFrame([{"metric": k, "value": v} for k, v in debug.items()])
            st.markdown("### Supporting metrics")
            st.dataframe(debug_df, use_container_width=True, hide_index=True)

with tabs[2]:
    if "idea_map" not in result:
        render_empty_state("🧠", "Idea map unavailable", "Collective mode is required to extract clustered ideas.")
    else:
        idea_map = result.get("idea_map", {}) or {}
        idea_map_meta = result.get("idea_map_meta", {}) or {}
        main_ideas = idea_map.get("main_ideas", []) or []

        if not main_ideas:
            render_empty_state("💡", "No ideas extracted", "Try a longer brainstorming audio clip, ideally 30 to 90 seconds.")
        else:
            source = idea_map_meta.get("source", "unknown")
            st.caption(f"Ideas are extracted from the full transcript and filtered to hide conversational filler. Source: {source}.")
            st.markdown("### Main ideas")
            for index, idea in enumerate(main_ideas, start=1):
                title = idea.get("title", f"Idea {index}")
                summary = idea.get("summary", "")
                with st.expander(f"🧠 {title}", expanded=index == 1):
                    if summary:
                        st.write(summary)

                    sub_ideas = idea.get("sub_ideas", []) or []
                    if not sub_ideas:
                        st.caption("No sub-ideas available.")
                    else:
                        rows = []
                        for s in sub_ideas:
                            evidences = s.get("evidence", []) or []
                            rows.append(
                                {
                                    "sub_idea": s.get("text", ""),
                                    "speakers": ", ".join(s.get("speakers", []) or []),
                                    "evidence": " | ".join(evidences[:2]),
                                }
                            )
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tabs[3]:
    if "transcript" not in result:
        render_empty_state("📝", "Transcript unavailable", "Collective mode is required to render transcript details.")
    else:
        transcript = result.get("transcript", {}) or {}
        full_text = transcript.get("full_text", "") or ""
        segs = transcript.get("segments", []) or []

        if not full_text and not segs:
            render_empty_state("🔇", "Transcript is empty", "Check the faster-whisper installation and the audio quality.")
        else:
            st.markdown("### Full transcript")
            st.text_area("Transcript preview", value=full_text[:8000], height=260, label_visibility="collapsed")

            st.markdown("### Timestamped segments")
            if segs:
                st.dataframe(pd.DataFrame(segs), use_container_width=True, hide_index=True)
            else:
                st.caption("No timestamped transcript segments available.")

with tabs[4]:
    if not segments:
        render_empty_state("⏱️", "No timeline data", "No diarization segments were returned for this analysis.")
    else:
        df = pd.DataFrame(segments)
        df["start_mmss"] = df["start"].apply(seconds_to_mmss)
        df["end_mmss"] = df["end"].apply(seconds_to_mmss)

        st.markdown("### Speaker segments")
        st.dataframe(df[["speaker", "start", "end", "start_mmss", "end_mmss"]], use_container_width=True, hide_index=True)

        detail_left, detail_right = st.columns(2, gap="large")
        with detail_left:
            st.markdown("### Overlaps")
            if overlaps:
                st.dataframe(pd.DataFrame(overlaps), use_container_width=True, hide_index=True)
            else:
                st.caption("No overlaps detected with the current thresholds.")
        with detail_right:
            st.markdown("### Interruptions")
            if interruptions:
                st.dataframe(pd.DataFrame(interruptions), use_container_width=True, hide_index=True)
            else:
                st.caption("No interruptions detected with the current thresholds.")

with tabs[5]:
    if not segments:
        render_empty_state("📊", "No chart data", "Run analysis with valid diarization output to generate charts.")
    else:
        df_chart = compute_speaking_and_interruptions(segments, interruptions)

        if df_chart.empty:
            render_empty_state("📉", "Not enough data", "There is not enough speaker timing information to draw charts.")
        else:
            st.markdown("### Speaking vs interruption time")
            df_chart = df_chart.sort_values("speaker").reset_index(drop=True)
            speakers_list = df_chart["speaker"].tolist()
            x = np.arange(len(speakers_list))
            width = 0.38

            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(10, 4.5))
            fig.patch.set_alpha(0)
            ax.set_facecolor("#111827")

            palette = ["#8b5cf6", "#22c55e", "#38bdf8", "#f59e0b", "#ef4444", "#14b8a6"]
            colors = [palette[i % len(palette)] for i in range(len(speakers_list))]

            ax.bar(
                x - width / 2,
                df_chart["speaking_min"].values,
                width,
                label="Speaking (min)",
                color=colors,
                alpha=0.92,
            )
            ax.bar(
                x + width / 2,
                df_chart["interruption_min"].values,
                width,
                label="Interrupting (min)",
                color=colors,
                alpha=0.38,
                hatch="//",
            )

            ax.set_xticks(x)
            ax.set_xticklabels(speakers_list, rotation=20, ha="right")
            ax.set_ylabel("Minutes")
            ax.set_title("Per-speaker speaking time and interruption time")
            ax.grid(axis="y", alpha=0.2, linestyle="--")
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.legend(frameon=False)

            st.pyplot(fig, clear_figure=True)
            st.dataframe(df_chart, use_container_width=True, hide_index=True)

with tabs[6]:
    show_json_collapsible("Full response JSON", result)
