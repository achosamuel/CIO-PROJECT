import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import io
import os

#python -m streamlit run frontend_app.py
# --- Page Configuration ---
st.set_page_config(
    page_title="Collective Intelligence Audio Analyzer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Professional, High-Contrast Design (Option 2 Sidebar) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Reset & Base */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Background & Container */
.main {
    background-color: #f8fafc;
    color: #1e293b;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 5rem;
    max-width: 1200px;
}

/* --- Sidebar Styling (Option 2: Glassmorphic Slate) --- */
section[data-testid="stSidebar"] {
    background-color: #f1f5f9 !important; /* Soft Slate Background */
    border-right: 1px solid #e2e8f0;
}

/* Sidebar Section Titles */
.sidebar-header {
    font-size: 0.75rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
    margin-top: 2rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e2e8f0;
}

/* Sidebar Inputs & Labels */
section[data-testid="stSidebar"] label {
    color: #334155 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

section[data-testid="stSidebar"] .stTextInput input {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    color: #1e293b !important;
}

section[data-testid="stSidebar"] .stSlider > div > div > div {
    background-color: #4f46e5 !important;
}

/* Sidebar Info Box */
.sidebar-info-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    margin-top: 2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.sidebar-info-card p {
    font-size: 0.85rem;
    color: #475569;
    line-height: 1.5;
    margin: 0;
}

/* --- Header & Titles --- */
.hero-title {
    text-align: center;
    padding: 40px 0 20px 0;
    margin-bottom: 20px;
}
.hero-title h1 {
    font-size: 3rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.03em;
    margin-bottom: 16px;
    background: linear-gradient(135deg, #0f172a 0%, #4f46e5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-title p {
    color: #64748b;
    font-size: 1.1rem;
    max-width: 700px;
    margin: 0 auto;
    line-height: 1.6;
}

/* Metric Cards Overwrite */
div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1) !important;
    border-color: #cbd5e1 !important;
}
div[data-testid="stMetric"] label {
    color: #64748b !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-size: 2.25rem !important;
    font-weight: 800 !important;
}

/* Custom Card Classes */
.glass-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 32px;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.04);
    margin-bottom: 24px;
}

/* Idea Map Styling */
.idea-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 6px solid #6366f1;
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
    transition: all 0.2s ease;
}
.idea-card:hover {
    border-color: #cbd5e1;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
}
.idea-card h3 {
    color: #1e293b;
    margin: 0 0 12px 0;
    font-size: 1.25rem;
    font-weight: 700;
}
.idea-card p {
    color: #475569;
    margin: 0;
    font-size: 1rem;
    line-height: 1.7;
}

/* Speaker Tags & Badges */
.speaker-tag {
    display: inline-flex;
    align-items: center;
    background: #f1f5f9;
    color: #475569;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 8px;
    border: 1px solid #e2e8f0;
}
.speaker-tag.primary {
    background: #eef2ff;
    color: #4f46e5;
    border-color: #c7d2fe;
}

/* Transcript Styling */
.transcript-line {
    padding: 16px 20px;
    border-radius: 12px;
    margin-bottom: 12px;
    background: #ffffff;
    border: 1px solid #f1f5f9;
    transition: all 0.2s;
}
.transcript-line:hover {
    background: #f8fafc;
    border-color: #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
}
.transcript-time {
    color: #94a3b8;
    font-size: 0.8rem;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-weight: 500;
    margin-right: 12px;
}
.transcript-speaker {
    font-weight: 700;
    color: #334155;
    margin-right: 8px;
}
.transcript-text {
    color: #475569;
    line-height: 1.6;
}

/* Tabs Styling */
button[data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 16px 24px !important;
    font-size: 1rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #4f46e5 !important;
    border-bottom: 3px solid #4f46e5 !important;
}

/* Interruption Stats */
.stat-highlight {
    background: #f8fafc;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    border: 1px solid #e2e8f0;
}
.stat-value {
    font-size: 1.75rem;
    font-weight: 800;
    color: #0f172a;
    display: block;
}
.stat-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* File Uploader Customization */
div[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 2px dashed #cbd5e1;
    border-radius: 20px;
    padding: 40px;
    transition: all 0.2s;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #6366f1;
    background: #f5f3ff;
}

/* Buttons */
.stButton > button {
    background: #4f46e5 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #4338ca !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.3) !important;
}

/* Score Label */
.score-label {
    text-align: center;
    color: #475569;
    font-size: 1rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: -10px;
}

/* Custom Expander */
details[data-testid="stExpander"] {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    margin-top: 12px !important;
}
details[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: #334155 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# --- Constants & Backend Config ---
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"

# --- Sidebar (Option 2 Implementation) ---
with st.sidebar:
    st.markdown('<div class="sidebar-header">Backend Config</div>', unsafe_allow_html=True)
    backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND_URL, help="The URL of your analysis engine.")
    
    st.markdown('<div class="sidebar-header">Parameters</div>', unsafe_allow_html=True)
    min_overlap = st.slider("Min Overlap (s)", 0.0, 1.0, 0.2, 0.05, help="Minimum overlap to count as an interruption.")
    cut_in_window = st.slider("Cut-in Window (s)", 0.1, 5.0, 1.0, 0.1, help="Window to analyze speaker transitions.")
    
    st.markdown('<div class="sidebar-header">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-info-card">
        <p>Analyzes meeting audio to extract <strong>collective intelligence</strong> metrics, speaker diarization, and an AI-powered idea map.</p>
        <hr style="margin: 12px 0; border-color: #f1f5f9;">
        <span style="font-size: 0.75rem; color: #94a3b8; font-weight: 500;">
            Powered by pyannote, faster-whisper, and Llama 3.3.
        </span>
    </div>
    """, unsafe_allow_html=True)

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
        response = requests.post(f"{backend_url}/collective", files=files, data=data, timeout=1000)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
        return None

# --- Plotly 
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#475569", family="Inter"),
    margin=dict(l=40, r=40, t=50, b=40),
)

# --- Hero Header ---
st.markdown("""
<div class="hero-title">
    <h1>Collective Intelligence Analyzer</h1>
    <p>Upload meeting audio to uncover speaker dynamics, participation balance, and AI-synthesized idea maps.</p>
</div>
""", unsafe_allow_html=True)

# --- Audio Input Section ---
col_upload_l, col_upload_c, col_upload_r = st.columns([1, 4, 1])
with col_upload_c:
    source_tabs = st.tabs(["Upload Audio", "Record in App"])

    with source_tabs[0]:
        uploaded_file = st.file_uploader(
            "Drop your audio file here",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
            help="Supported: WAV, MP3, M4A, FLAC, OGG"
        )
        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.getvalue()
            st.audio(uploaded_bytes, format=uploaded_file.type or "audio/wav")
            st.markdown("")
            if st.button("Run Deep Analysis", type="primary", use_container_width=True):
                with st.spinner("Analyzing audio..."):
                    results = call_analyze(uploaded_bytes, uploaded_file.name, min_overlap, cut_in_window)
                    if results:
                        st.session_state['analysis_results'] = results
                        st.success("Analysis complete!")

    with source_tabs[1]:
        if hasattr(st, "audio_input"):
            recorded_clip = st.audio_input("Record audio directly")
            if recorded_clip is not None:
                recorded_bytes = recorded_clip.getvalue()
                st.audio(recorded_bytes, format=recorded_clip.type or "audio/wav")
                st.markdown("")
                if st.button("Analyze Recorded Audio", type="primary", use_container_width=True):
                    with st.spinner("Analyzing audio..."):
                        filename = recorded_clip.name or "recorded_audio.wav"
                        results = call_analyze(recorded_bytes, filename, min_overlap, cut_in_window)
                        if results:
                            st.session_state['analysis_results'] = results
                            st.success("Analysis complete!")
        else:
            st.warning("Your Streamlit version does not support in-app recording.")

# --- Results Display ---
if 'analysis_results' in st.session_state:
    res = st.session_state['analysis_results']
    tab_overview, tab_ideas, tab_speakers, tab_transcript = st.tabs([
        "Overview", "Idea Map", "Speakers", "Transcript"
    ])

    # 1. Overview & Scores
    with tab_overview:
        st.markdown("<br>", unsafe_allow_html=True)
        scores = res.get("scores", {})
        overall = scores.get('collective_organization', 0)
        
        col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
        with col_g2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=overall,
                number=dict(suffix="%", font=dict(size=56, color="#1e293b", weight="bold")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#94a3b8", tickfont=dict(color="#64748b")),
                    bar=dict(color="#6366f1"),
                    bgcolor="#f1f5f9",
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 40], color="#fee2e2"),
                        dict(range=[40, 70], color="#fef3c7"),
                        dict(range=[70, 100], color="#dcfce7"),
                    ],
                ),
            ))
            # Updated layout to use only the PLOTLY_LAYOUT which already contains margins
            fig_gauge.update_layout(**PLOTLY_LAYOUT, height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown('<div class="score-label">Collective Organization Score</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Interruption Rate", f"{scores.get('independence', 0)}%")
        col2.metric("Participation Balance", f"{scores.get('participation_balance', 0)}%")
        col3.metric("Idea Diversity", f"{scores.get('idea_diversity', 0)}%")

        st.markdown("<br>", unsafe_allow_html=True)
        # Radar Chart
        categories = ['Interruption Rate', 'Participation Balance', 'Idea Diversity']
        values = [scores.get('independence', 0), scores.get('participation_balance', 0), scores.get('idea_diversity', 0)]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(99, 102, 241, 0.1)',
            line=dict(color='#6366f1', width=3),
            marker=dict(size=10, color='#4f46e5'),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="#e2e8f0", tickfont=dict(color="#94a3b8")),
                angularaxis=dict(gridcolor="#e2e8f0", tickfont=dict(color="#475569", size=12)),
            ),
            showlegend=False, height=450, **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # 2. Idea Map
    with tab_ideas:
        st.markdown("<br>", unsafe_allow_html=True)
        idea_map = res.get("idea_map", {})
        main_ideas = idea_map.get("main_ideas", [])
        if not main_ideas:
            st.warning("No significant ideas were extracted.")
        else:
            for idea in main_ideas:
                st.markdown(f"""
                <div class="idea-card">
                    <h3>{idea.get('title', 'Untitled Idea')}</h3>
                    <p>{idea.get('summary', '')}</p>
                </div>
                """, unsafe_allow_html=True)
                for sub in idea.get("sub_ideas", []):
                    with st.expander(f"{sub.get('text', '')[:100]}..."):
                        speakers = sub.get("speakers", [])
                        if speakers:
                            tags = " ".join([f'<span class="speaker-tag primary">{s}</span>' for s in speakers])
                            st.markdown(f"**Contributors:** {tags}", unsafe_allow_html=True)
                        evidence = sub.get("evidence", [])
                        if evidence:
                            st.markdown("**Evidence:**")
                            for e in evidence:
                                st.markdown(f"> _{e}_")

    # 3. Speaker Analysis
    with tab_speakers:
        st.markdown("<br>", unsafe_allow_html=True)
        debug_info = res.get("scores", {}).get("debug", {})
        speaker_name_map = res.get("speaker_name_map", {})
        raw_speaking_times = debug_info.get("speaking_time_seconds", {})
        speaking_times = {speaker_name_map.get(k, k): v for k, v in raw_speaking_times.items()}

        if speaking_times:
            col_left, col_right = st.columns([1.2, 1])
            with col_left:
                df_speakers = pd.DataFrame([{"Speaker": s, "Seconds": t} for s, t in speaking_times.items()])
                # Using a High-Contrast Qualitative Palette (Set1) for better distinction
                fig_pie = px.pie(df_speakers, values='Seconds', names='Speaker', hole=0.5,
                                color_discrete_sequence=px.colors.qualitative.Set1)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#fff', width=2)))
                fig_pie.update_layout(height=450, **PLOTLY_LAYOUT, title="Speaking Time Distribution")
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_right:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### Interruption Analysis")
                interruption_count = len(res.get('interruptions', []))
                ipm = debug_info.get('interruptions_per_min', 0)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f'<div class="stat-highlight"><span class="stat-value">{interruption_count}</span><span class="stat-label">Total</span></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="stat-highlight"><span class="stat-value">{ipm:.1f}</span><span class="stat-label">Per Min</span></div>', unsafe_allow_html=True)
                
                if res.get('interruptions'):
                    st.markdown("<br>", unsafe_allow_html=True)
                    df_int = pd.DataFrame(res.get('interruptions'))
                    st.dataframe(df_int[['interrupter', 'interrupted', 'timestamp']], use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Timeline
        st.markdown("### Speaker Timeline")
        segments = res.get("segments", [])
        if segments:
            df_timeline = pd.DataFrame(segments)
            epoch = pd.Timestamp("2000-01-01")
            df_timeline["start_dt"] = df_timeline["start"].apply(lambda s: epoch + pd.Timedelta(seconds=s))
            df_timeline["end_dt"] = df_timeline["end"].apply(lambda s: epoch + pd.Timedelta(seconds=s))
            # High-contrast palette for timeline as well
            fig_timeline = px.timeline(df_timeline, x_start="start_dt", x_end="end_dt", y="speaker", color="speaker",
                                       color_discrete_sequence=px.colors.qualitative.Set1)
            fig_timeline.update_layout(height=300, xaxis_title="Time", yaxis_title="", showlegend=False, **PLOTLY_LAYOUT)
            fig_timeline.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_timeline, use_container_width=True)

    # 4. Transcript
    with tab_transcript:
        st.markdown("<br>", unsafe_allow_html=True)
        search_query = st.text_input("Search transcript...", "", placeholder="Filter by keyword or speaker...")
        utterances = res.get("speaker_utterances", [])
        for u in utterances:
            speaker = u.get("speaker", "UNKNOWN")
            text = u.get("text", "")
            timestamp = format_time(u.get("start", 0))
            if search_query.lower() in text.lower() or search_query.lower() in speaker.lower():
                st.markdown(f"""
                <div class="transcript-line">
                    <span class="transcript-time">[{timestamp}]</span>
                    <span class="transcript-speaker">{speaker}</span>
                    <span class="transcript-text">{text}</span>
                </div>
                """, unsafe_allow_html=True)

else:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_e1, col_e2, col_e3 = st.columns([1, 2, 1])
    with col_e2:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 60px 40px;">
            <div style="font-size: 4rem; margin-bottom: 24px;"></div>
            <h3 style="color: #0f172a; font-weight: 700; margin-bottom: 12px;">Ready for Analysis</h3>
            <p style="color: #64748b;">Upload a meeting recording to extract intelligence metrics and speaker insights.</p>
        </div>
        """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding-bottom: 40px;">
    <span style="color: #94a3b8; font-size: 0.9rem; font-weight: 500;">
        Powered by <strong style="color:#4f46e5;">pyannote</strong> · 
        <strong style="color:#4f46e5;">faster-whisper</strong> · 
        <strong style="color:#4f46e5;">Llama 3.3</strong>
    </span>
</div>
""", unsafe_allow_html=True)
