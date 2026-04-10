import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import io
import os
#python -m streamlit run lovable_app.py
# --- Page Configuration ---
st.set_page_config(
    page_title="Collective Intelligence Audio Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Premium Design ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
    color: #e0e0e0;
}
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* Hide default header bar */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a3e 0%, #0f0c29 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] label {
    color: #c0c0d0 !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.70);
    border: 1px solid rgba(148,163,184,0.25);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(12px);
    transition: transform 0.2s, box-shadow 0.2s;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
}
div[data-testid="stMetric"] label {
    color: #475569 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #334155 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    background: transparent !important;
    color: #9ca3af !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-weight: 500;
    padding: 12px 20px !important;
    transition: all 0.2s;
}
button[data-baseweb="tab"]:hover {
    color: #c7d2fe !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #818cf8 !important;
    border-bottom: 2px solid #818cf8 !important;
}

/* Glass card helper */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(12px);
    margin-bottom: 16px;
}

/* Idea cards */
.idea-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 4px solid #818cf8;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 16px;
    backdrop-filter: blur(12px);
    transition: transform 0.2s, box-shadow 0.2s;
}
.idea-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.12);
}
.idea-card h3 {
    color: #e0e7ff;
    margin: 0 0 8px 0;
    font-size: 1.1rem;
    font-weight: 600;
}
.idea-card p {
    color: #9ca3af;
    margin: 0;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Speaker tag */
.speaker-tag {
    display: inline-block;
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.2));
    color: #c7d2fe;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.8em;
    margin-right: 6px;
    border: 1px solid rgba(139,92,246,0.25);
    font-weight: 500;
}

/* Transcript */
.transcript-line {
    padding: 10px 16px;
    border-radius: 8px;
    margin-bottom: 6px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    transition: background 0.15s;
}
.transcript-line:hover {
    background: rgba(255,255,255,0.05);
}
.transcript-speaker {
    font-weight: 600;
    color: #818cf8;
}
.transcript-time {
    color: #6b7280;
    font-size: 0.8em;
    font-family: 'SF Mono', 'Fira Code', monospace;
}

/* File uploader */
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border: 2px dashed rgba(129, 140, 248, 0.3);
    border-radius: 16px;
    padding: 20px;
    transition: border-color 0.2s;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(129, 140, 248, 0.6);
}

/* Primary button */
.stButton > button[kind="primary"],
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 32px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.02em;
    transition: all 0.2s !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(99, 102, 241, 0.45) !important;
}

/* Expander */
details[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
}
details[data-testid="stExpander"] summary {
    color: #c7d2fe !important;
    font-weight: 500;
}

/* Plotly chart background */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.06) !important;
}

/* Success / Info / Warning boxes */
div[data-testid="stAlert"] {
    border-radius: 12px;
    border: none;
}

/* Hero title */
.hero-title {
    text-align: center;
    padding: 20px 0 10px 0;
}
.hero-title h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}
.hero-title p {
    color: #9ca3af;
    font-size: 1.05rem;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
}

/* Score gauge label */
.score-label {
    text-align: center;
    color: #9ca3af;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 8px;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(129, 140, 248, 0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# --- Constants & Backend Config ---
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    backend_url = st.text_input("🔗 Backend URL", value=DEFAULT_BACKEND_URL)
    st.markdown("")
    min_overlap = st.slider("🔀 Min Overlap Threshold (s)", 0.0, 1.0, 0.2, 0.05)
    cut_in_window = st.slider("✂️ Cut-in Window (s)", 0.1, 5.0, 1.0, 0.1)
    st.markdown("---")
    st.markdown("### 📖 About")
    st.info(
        "Analyzes meeting audio to extract collective intelligence metrics, "
        "speaker diarization, and an AI-powered idea map."
    )
    st.markdown("")
    st.caption("Built with pyannote · faster-whisper · Llama 3.3")


# --- Helper Functions ---
def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def call_analyze(file_bytes, filename, min_overlap, cut_in):
    files = {"file": (filename, file_bytes)}
    data = {
        "min_overlap_threshold": min_overlap,
        "cut_in_window": cut_in
    }
    try:
        response = requests.post(f"{backend_url}/collective", files=files, data=data, timeout=300)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"❌ Error connecting to backend: {e}")
        return None


# --- Plotly dark theme ---
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c0c0d0", family="Inter"),
    margin=dict(l=40, r=40, t=50, b=40),
)


# --- Hero Header ---
st.markdown("""
<div class="hero-title">
    <h1>🧠 Collective Intelligence Analyzer</h1>
    <p>Upload a meeting recording to uncover speaker dynamics, extract key ideas, and measure collective intelligence.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# --- Upload Section ---
col_upload_l, col_upload_c, col_upload_r = st.columns([1, 3, 1])
with col_upload_c:
    uploaded_file = st.file_uploader(
        "Drop your audio file here",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="Supported: WAV, MP3, M4A, FLAC, OGG"
    )
    if uploaded_file is not None:
        st.markdown("")
        if st.button("🚀 Run Deep Analysis", type="primary", use_container_width=True):
            with st.spinner("🔬 Analyzing audio — diarization, transcription & LLM processing..."):
                file_bytes = uploaded_file.read()
                results = call_analyze(file_bytes, uploaded_file.name, min_overlap, cut_in_window)
                if results:
                    st.session_state['analysis_results'] = results
                    st.success("✅ Analysis complete!")

st.markdown("")

# --- Results Display ---
if 'analysis_results' in st.session_state:
    res = st.session_state['analysis_results']

    tab_overview, tab_ideas, tab_speakers, tab_transcript = st.tabs([
        "📊 Overview",
        "💡 Idea Map",
        "👥 Speakers",
        "📝 Transcript"
    ])

    # ──────────────────────────────────
    # 1. Overview & Scores
    # ──────────────────────────────────
    with tab_overview:
        st.markdown("")
        scores = res.get("scores", {})

        # Main score gauge
        overall = scores.get('collective_organization', 0)
        col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
        with col_g2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=overall,
                number=dict(suffix="/100", font=dict(size=48, color="#312e81")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#4b5563", tickfont=dict(color="#6b7280")),
                    bar=dict(color="#818cf8"),
                    bgcolor="rgba(255,255,255,0.03)",
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 40], color="rgba(239,68,68,0.15)"),
                        dict(range=[40, 70], color="rgba(251,191,36,0.15)"),
                        dict(range=[70, 100], color="rgba(52,211,153,0.15)"),
                    ],
                    threshold=dict(line=dict(color="#f472b6", width=3), thickness=0.8, value=overall),
                ),
            ))
            layout_gauge = {
                **PLOTLY_LAYOUT,
                "height": 280,
                "margin": dict(l=30, r=30, t=30, b=10),
            }

            fig_gauge.update_layout(**layout_gauge)
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown('<div class="score-label">Collective Organization Score</div>', unsafe_allow_html=True)

        st.markdown("")

        # Sub-metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("🛡️ Independence", f"{scores.get('independence', 0)}%")
        col2.metric("⚖️ Participation Balance", f"{scores.get('participation_balance', 0)}%")
        col3.metric("🌈 Idea Diversity", f"{scores.get('idea_diversity', 0)}%")

        st.markdown("")

        # Radar
        categories = ['Independence', 'Participation\nBalance', 'Idea\nDiversity']
        values = [
            scores.get('independence', 0),
            scores.get('participation_balance', 0),
            scores.get('idea_diversity', 0),
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(129, 140, 248, 0.15)',
            line=dict(color='#818cf8', width=2),
            marker=dict(size=8, color='#c084fc'),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.06)",
                                tickfont=dict(color="#6b7280", size=10)),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.06)",
                                 tickfont=dict(color="#c0c0d0", size=12)),
            ),
            showlegend=False,
            height=380,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ──────────────────────────────────
    # 2. Idea Map
    # ──────────────────────────────────
    with tab_ideas:
        st.markdown("")
        idea_map = res.get("idea_map", {})
        main_ideas = idea_map.get("main_ideas", [])

        if not main_ideas:
            st.warning("No significant ideas were extracted.")
        else:
            for idx, idea in enumerate(main_ideas):
                st.markdown(f"""
                <div class="idea-card">
                    <h3>💡 {idea.get('title', 'Untitled Idea')}</h3>
                    <p>{idea.get('summary', '')}</p>
                </div>
                """, unsafe_allow_html=True)

                for sub in idea.get("sub_ideas", []):
                    with st.expander(f"🔹 {sub.get('text', '')[:100]}"):
                        speakers = sub.get("speakers", [])
                        if speakers:
                            tags = " ".join([f'<span class="speaker-tag">{s}</span>' for s in speakers])
                            st.markdown(f"**Contributors:** {tags}", unsafe_allow_html=True)
                        evidence = sub.get("evidence", [])
                        if evidence:
                            st.markdown("**Evidence:**")
                            for e in evidence:
                                st.markdown(f"> _{e}_")

                if idx < len(main_ideas) - 1:
                    st.markdown("")

    # ──────────────────────────────────
    # 3. Speaker Analysis
    # ──────────────────────────────────
    with tab_speakers:
        st.markdown("")
        debug_info = res.get("scores", {}).get("debug", {})
        speaker_name_map = res.get("speaker_name_map", {})
        raw_speaking_times = debug_info.get("speaking_time_seconds", {})
        speaking_times = {}
        for speaker_label, seconds in raw_speaking_times.items():
            display_name = speaker_name_map.get(speaker_label, speaker_label)
            speaking_times[display_name] = speaking_times.get(display_name, 0) + seconds

        if speaking_times:
            df_speakers = pd.DataFrame([
                {"Speaker": s, "Seconds": t, "Minutes": round(t / 60, 2)}
                for s, t in speaking_times.items()
            ])

            col_left, col_right = st.columns(2)

            with col_left:
                colors = px.colors.qualitative.Pastel
                fig_pie = px.pie(
                    df_speakers, values='Seconds', names='Speaker',
                    title="Speaking Time Distribution",
                    color_discrete_sequence=colors,
                    hole=0.45,
                )
                fig_pie.update_traces(
                    textposition='inside', textinfo='percent+label',
                    textfont=dict(color="#1a1a3e", size=12),
                    marker=dict(line=dict(color='rgba(0,0,0,0.2)', width=1)),
                )
                fig_pie.update_layout(height=400, **PLOTLY_LAYOUT)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_right:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### ⚡ Interruption Analysis")
                interruption_count = len(res.get('interruptions', []))
                ipm = debug_info.get('interruptions_per_min', 0)

                c1, c2 = st.columns(2)
                c1.metric("Total", interruption_count)
                c2.metric("Per Minute", f"{ipm:.1f}")

                if res.get('interruptions'):
                    df_int = pd.DataFrame(res.get('interruptions'))
                    st.dataframe(
                        df_int[['interrupter', 'interrupted', 'timestamp']],
                        use_container_width=True, hide_index=True,
                    )
                st.markdown('</div>', unsafe_allow_html=True)

        # Timeline
        st.markdown("")
        st.markdown("### 📅 Speaker Timeline")
        segments = res.get("segments", [])
        if segments:
            df_timeline = pd.DataFrame(segments)
            # Convert seconds to datetime for px.timeline
            epoch = pd.Timestamp("2000-01-01")
            df_timeline["start_dt"] = df_timeline["start"].apply(lambda s: epoch + pd.Timedelta(seconds=s))
            df_timeline["end_dt"] = df_timeline["end"].apply(lambda s: epoch + pd.Timedelta(seconds=s))

            palette = ["#818cf8", "#c084fc", "#f472b6", "#34d399", "#fbbf24", "#60a5fa", "#f87171"]
            fig_timeline = px.timeline(
                df_timeline, x_start="start_dt", x_end="end_dt",
                y="speaker", color="speaker",
                color_discrete_sequence=palette,
            )
            fig_timeline.update_layout(
                height=max(200, len(speaking_times) * 60 + 80),
                xaxis_title="Time",
                yaxis_title="",
                showlegend=False,
                **PLOTLY_LAYOUT,
            )
            fig_timeline.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_timeline, use_container_width=True)

    # ──────────────────────────────────
    # 4. Transcript
    # ──────────────────────────────────
    with tab_transcript:
        st.markdown("")
        search_query = st.text_input("🔍 Search transcript...", "", placeholder="Type to filter...")
        st.markdown("")

        utterances = res.get("speaker_utterances", [])
        found = 0
        for u in utterances:
            speaker = u.get("speaker", "UNKNOWN")
            text = u.get("text", "")
            timestamp = format_time(u.get("start", 0))

            if search_query and search_query.lower() not in text.lower() and search_query.lower() not in speaker.lower():
                continue

            found += 1
            st.markdown(f"""
            <div class="transcript-line">
                <span class="transcript-time">[{timestamp}]</span>
                <span class="transcript-speaker">{speaker}:</span>
                <span style="color: #d1d5db;">{text}</span>
            </div>
            """, unsafe_allow_html=True)

        if search_query and found == 0:
            st.info("No matching utterances found.")

else:
    # Empty state
    st.markdown("")
    col_e1, col_e2, col_e3 = st.columns([1, 2, 1])
    with col_e2:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 60px 40px;">
            <div style="font-size: 4rem; margin-bottom: 16px;">🎙️</div>
            <h3 style="color: #e0e7ff; font-weight: 600; margin-bottom: 8px;">No analysis yet</h3>
            <p style="color: #9ca3af;">Upload an audio file above and click <strong>Run Deep Analysis</strong> to get started.</p>
        </div>
        """, unsafe_allow_html=True)


# --- Footer ---
st.markdown("")
st.divider()
st.markdown("""
<div style="text-align: center; padding: 10px 0;">
    <span style="color: #6b7280; font-size: 0.85rem;">
        Powered by <strong style="color:#818cf8;">pyannote</strong> · 
        <strong style="color:#c084fc;">faster-whisper</strong> · 
        <strong style="color:#f472b6;">Llama 3.3 via Groq</strong>
    </span>
</div>
""", unsafe_allow_html=True)
