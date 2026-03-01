"""
FastAPI backend for diarization + collective organization + idea map (FREE).

Pipeline:
- Normalize audio to WAV 16k mono (utils_audio.py)
- Speaker diarization (pyannote) using HF_TOKEN
- Transcription (offline) using faster-whisper
- Idea map (offline) using TF-IDF + clustering (no torch / no transformers)

Run (PowerShell):
  $env:HF_TOKEN="hf_iRzVdteghGHWYljbOQYAmPOJaYJM"
  uvicorn backend_app:app --host 127.0.0.1 --port 8000 --reload

Notes:
- Requires FFmpeg available on PATH (used by utils_audio conversion).
- Requires you accepted the model conditions for pyannote/speaker-diarization-3.1 on Hugging Face.
"""

from __future__ import annotations

import inspect
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

from utils_audio import AudioConversionError, convert_audio_bytes_to_wav16k_mono


# -------------------------
# App + CORS
# -------------------------

app = FastAPI(title="Collective Intelligence Audio Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ALLOWED_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

# Caches (loaded once, reused)
_PIPELINE: Any | None = None
_WHISPER_MODEL = None


# -------------------------
# Data structures
# -------------------------

@dataclass
class Segment:
    start: float
    end: float
    speaker: str


# -------------------------
# Utils
# -------------------------

def _require_positive(name: str, value: float) -> None:
    if value < 0:
        raise HTTPException(status_code=400, detail=f"{name} must be >= 0")


def _safe_suffix(filename: str | None) -> str:
    return Path(filename or "").suffix.lower()


# -------------------------
# HF + pyannote diarization
# -------------------------

def get_diarization_pipeline() -> Any:
    """
    Load/cached pyannote pipeline with HF token.

    This tries to be robust across pyannote/hub versions by:
    - Passing auth token directly to Pipeline.from_pretrained when supported.
    - Falling back to huggingface_hub.login if needed.
    """
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    token = os.getenv("HF_TOKEN", "").strip()
    if not token or not token.startswith("hf_"):
        raise HTTPException(
            status_code=503,
            detail=(
                "Missing/invalid HF_TOKEN. Set HF_TOKEN (starts with 'hf_') and accept the "
                "terms for pyannote/speaker-diarization-3.1 on Hugging Face."
            ),
        )

    try:
        from pyannote.audio import Pipeline

        # Some versions support: Pipeline.from_pretrained(..., token=...)
        # Others: use_auth_token=...
        sig = inspect.signature(Pipeline.from_pretrained)
        if "token" in sig.parameters:
            _PIPELINE = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
        elif "use_auth_token" in sig.parameters:
            _PIPELINE = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        else:
            # Last resort: login then load
            from huggingface_hub import login
            login(token=token)
            _PIPELINE = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

        if _PIPELINE is None:
            raise RuntimeError("Pipeline.from_pretrained returned None")

        return _PIPELINE

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load diarization model: {exc}",
        ) from exc


# -------------------------
# Overlap + interruption logic
# -------------------------

def merge_overlap_events(overlap_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge overlap windows that intersect/touch and combine speaker sets."""
    if not overlap_events:
        return []

    overlap_events = sorted(overlap_events, key=lambda x: (x["start"], x["end"]))
    merged: list[dict[str, Any]] = []

    for event in overlap_events:
        if not merged or event["start"] > merged[-1]["end"]:
            merged.append(
                {"start": event["start"], "end": event["end"], "speakers": set(event["speakers"])}
            )
        else:
            merged[-1]["end"] = max(merged[-1]["end"], event["end"])
            merged[-1]["speakers"].update(event["speakers"])

    out = []
    for event in merged:
        speakers_sorted = sorted(event["speakers"])
        out.append(
            {
                "start": round(float(event["start"]), 3),
                "end": round(float(event["end"]), 3),
                "speakers": speakers_sorted,
                "duration": round(float(event["end"]) - float(event["start"]), 3),
            }
        )
    return out


def detect_overlaps(segments: list[Segment], min_overlap: float = 0.2) -> list[dict[str, Any]]:
    """Find overlap windows across different-speaker segments."""
    events: list[dict[str, Any]] = []
    n = len(segments)

    for i in range(n):
        a = segments[i]
        for j in range(i + 1, n):
            b = segments[j]
            if a.speaker == b.speaker:
                continue
            start = max(a.start, b.start)
            end = min(a.end, b.end)
            if end > start and (end - start) >= min_overlap:
                events.append({"start": start, "end": end, "speakers": [a.speaker, b.speaker]})

    return merge_overlap_events(events)


def infer_interruptions(
    segments: list[Segment],
    overlaps: list[dict[str, Any]],
    cut_in_window: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Infer interruptions from overlap windows.

    Heuristic:
    - interrupted speaker: already speaking just before overlap start
    - interrupter: starts within cut_in_window of overlap start and is active at overlap start
    """
    interruptions: list[dict[str, Any]] = []

    for overlap in overlaps:
        o_start = float(overlap["start"])
        o_end = float(overlap["end"])

        interrupted: Optional[str] = None
        interrupter: Optional[str] = None

        for seg in segments:
            if seg.start <= o_start <= seg.end and seg.start <= (o_start - 0.05):
                interrupted = seg.speaker

        for seg in segments:
            starts_near = 0 <= (o_start - seg.start) <= cut_in_window
            active = seg.start <= o_start <= seg.end
            if starts_near and active and seg.speaker != interrupted:
                interrupter = seg.speaker
                break

        if interrupted and interrupter and interrupted != interrupter:
            interruptions.append(
                {
                    "start": round(o_start, 3),
                    "end": round(o_end, 3),
                    "interrupter": interrupter,
                    "interrupted": interrupted,
                    "overlap_duration": round(o_end - o_start, 3),
                }
            )

    # dedupe
    seen = set()
    out = []
    for it in interruptions:
        key = (it["start"], it["end"], it["interrupter"], it["interrupted"])
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


# -------------------------
# Diarization
# -------------------------

def diarize_wav(wav_path: Path, min_overlap_threshold: float, cut_in_window: float) -> dict[str, Any]:
    pipeline = get_diarization_pipeline()
    diarization = pipeline(str(wav_path))

    segments: list[Segment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(Segment(float(turn.start), float(turn.end), str(speaker)))

    segments.sort(key=lambda s: (s.start, s.end))
    speakers = sorted({s.speaker for s in segments})
    overlaps = detect_overlaps(segments, min_overlap=min_overlap_threshold)
    interruptions = infer_interruptions(segments, overlaps, cut_in_window=cut_in_window)
    audio_duration = max((s.end for s in segments), default=0.0)

    return {
        "segments_obj": segments,
        "num_speakers": len(speakers),
        "speakers": speakers,
        "segments": [
            {"start": round(s.start, 3), "end": round(s.end, 3), "speaker": s.speaker} for s in segments
        ],
        "overlaps": overlaps,
        "interruptions": interruptions,
        "audio_duration": audio_duration,
    }


# -------------------------
# Transcription (offline: faster-whisper)
# -------------------------

def get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

    model_size = os.getenv("WHISPER_MODEL_SIZE", "base").strip()  # base/small/medium...
    try:
        from faster_whisper import WhisperModel

        _WHISPER_MODEL = WhisperModel(model_size, device="cpu", compute_type="int8")
        return _WHISPER_MODEL
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load faster-whisper model. Install faster-whisper. Error: {exc}",
        ) from exc


def transcribe_with_whisper(wav_path: Path) -> dict[str, Any]:
    model = get_whisper_model()
    segments_iter, _info = model.transcribe(
        str(wav_path),
        language="en",
        vad_filter=True,
    )

    segments: list[dict[str, Any]] = []
    texts = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        segments.append(
            {"start": round(float(seg.start), 3), "end": round(float(seg.end), 3), "text": text}
        )
        texts.append(text)

    return {"full_text": " ".join(texts).strip(), "segments": segments}


# -------------------------
# Speaker-text alignment
# -------------------------

def align_speakers_to_transcript(
    diarization_segments: list[Segment],
    transcript_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign each transcript segment to the diarized speaker with max temporal overlap."""
    utterances: list[dict[str, Any]] = []

    for tseg in transcript_segments:
        t_start = float(tseg.get("start", 0.0))
        t_end = float(tseg.get("end", t_start))
        text = str(tseg.get("text", "")).strip()
        if not text:
            continue

        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for dseg in diarization_segments:
            overlap = max(0.0, min(t_end, dseg.end) - max(t_start, dseg.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dseg.speaker

        utterances.append(
            {"speaker": best_speaker, "start": round(t_start, 3), "end": round(t_end, 3), "text": text}
        )

    return utterances


# -------------------------
# Idea map (offline: TF-IDF + clustering)
# -------------------------

def split_into_idea_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    parts = [p for p in parts if len(p) >= 10]
    return parts[:300]  # safety cap


def cluster_ideas(sentences: list[str]) -> dict[str, Any]:
    """
    Cluster idea sentences into "main ideas" using TF-IDF + agglomerative clustering.
    This avoids torch/transformers completely.
    """
    if not sentences:
        return {"main_ideas": []}

    vec = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    X = vec.fit_transform(sentences)

    # smaller => more clusters
    dist_thresh = float(os.getenv("IDEA_CLUSTER_DISTANCE", "0.70"))
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=dist_thresh,
    )

    # Convert to dense for sklearn clustering (safe due to cap)
    labels = clustering.fit_predict(X.toarray()).tolist()

    clusters: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(i)

    cluster_items = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)

    main_ideas = []
    for ci, (_lab, idxs) in enumerate(cluster_items, start=1):
        # representative sentence = highest average similarity within cluster
        subX = X[idxs]
        sim = (subX @ subX.T).toarray()
        rep_i = idxs[int(np.argmax(sim.mean(axis=1)))]

        title = sentences[rep_i]
        if len(title) > 80:
            title = title[:77] + "..."

        cluster_texts = [sentences[i] for i in idxs]
        summary = cluster_texts[0]
        if len(cluster_texts) > 1 and len(summary) < 120:
            summary = (summary + " " + cluster_texts[1])[:250]

        sub_ideas = [
            {"text": sentences[i], "speakers": [], "evidence": [sentences[i][:120]]}
            for i in idxs
        ]

        main_ideas.append(
            {"id": f"I{ci}", "title": title, "summary": summary, "sub_ideas": sub_ideas[:12]}
        )

    return {"main_ideas": main_ideas[:12]}


def attach_speakers_to_ideas(idea_map: dict[str, Any], utterances: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Simple speaker attribution:
    - if the first 30 chars of a sub-idea appears in an utterance, attribute that speaker.
    """
    if not idea_map.get("main_ideas") or not utterances:
        return idea_map

    utt = [(u.get("speaker", "UNKNOWN"), (u.get("text", "") or "").lower()) for u in utterances]

    for idea in idea_map["main_ideas"]:
        for sub in idea.get("sub_ideas", []):
            t = (sub.get("text", "") or "").lower()
            if not t:
                continue
            key = t[:30]
            speakers = {spk for spk, txt in utt if key in txt}
            if speakers:
                sub["speakers"] = sorted(speakers)

    return idea_map


# -------------------------
# Scores
# -------------------------

def gini(values: list[float]) -> float:
    arr = [max(0.0, float(v)) for v in values]
    n = len(arr)
    if n == 0:
        return 0.0
    total = sum(arr)
    if total <= 1e-9:
        return 0.0
    arr.sort()
    cum = 0.0
    for i, v in enumerate(arr, start=1):
        cum += i * v
    return max(0.0, min(1.0, (2 * cum) / (n * total) - (n + 1) / n))


def compute_scores(
    overlaps: list[dict[str, Any]],
    interruptions: list[dict[str, Any]],
    segments: list[Segment],
    idea_map: dict[str, Any],
) -> dict[str, Any]:
    speaking_times: dict[str, float] = {}
    for seg in segments:
        speaking_times[seg.speaker] = speaking_times.get(seg.speaker, 0.0) + max(0.0, seg.end - seg.start)

    total_speaking_time = sum(speaking_times.values())
    total_overlap_time = sum(float(o.get("duration", 0.0)) for o in overlaps)
    overlap_ratio = (total_overlap_time / total_speaking_time) if total_speaking_time > 1e-9 else 0.0

    audio_duration = max((s.end for s in segments), default=0.0)
    interruptions_per_min = len(interruptions) / ((audio_duration / 60.0) + 1e-6)

    independence = 100.0
    independence -= overlap_ratio * 80.0
    independence -= min(30.0, interruptions_per_min * 10.0)
    independence = max(0.0, min(100.0, independence))

    g = gini(list(speaking_times.values()))
    participation_balance = max(0.0, min(100.0, (1.0 - g) * 100.0))

    num_main_ideas = len(idea_map.get("main_ideas", []) or [])
    idea_diversity = max(0.0, min(100.0, (num_main_ideas / 8.0) * 100.0))

    collective = round(0.4 * independence + 0.3 * participation_balance + 0.3 * idea_diversity, 1)

    return {
        "collective_organization": collective,
        "independence": round(independence, 1),
        "participation_balance": round(participation_balance, 1),
        "idea_diversity": round(idea_diversity, 1),
        "debug": {
            "overlap_ratio": round(overlap_ratio, 4),
            "interruptions_per_min": round(interruptions_per_min, 3),
            "speaking_time_seconds": {k: round(v, 3) for k, v in speaking_times.items()},
            "gini_speaking_time": round(g, 4),
            "num_main_ideas": num_main_ideas,
            "cluster_distance_threshold": float(os.getenv("IDEA_CLUSTER_DISTANCE", "0.70")),
        },
    }


# -------------------------
# Routes
# -------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/versions")
def versions() -> dict[str, Any]:
    out: dict[str, Any] = {"python": os.sys.version}

    def add(pkg: str):
        try:
            mod = __import__(pkg)
            out[pkg] = getattr(mod, "_version", "no __version_")
        except Exception as e:
            out[pkg] = f"IMPORT_ERROR: {type(e).__name__}: {e}"

    add("torch")
    add("huggingface_hub")
    add("pyannote")
    add("pyannote.audio")
    add("faster_whisper")
    add("sklearn")
    add("numpy")
    return out

@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    min_overlap_threshold: float = Form(0.2),
    cut_in_window: float = Form(1.0),
) -> dict[str, Any]:
    suffix = _safe_suffix(file.filename)
    if suffix and suffix not in _ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension '{suffix}'.")

    _require_positive("min_overlap_threshold", min_overlap_threshold)
    _require_positive("cut_in_window", cut_in_window)

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "normalized.wav"
            convert_audio_bytes_to_wav16k_mono(raw_bytes, wav_path)
            diarized = diarize_wav(wav_path, min_overlap_threshold, cut_in_window)

        return {
            "num_speakers": diarized["num_speakers"],
            "speakers": diarized["speakers"],
            "segments": diarized["segments"],
            "overlaps": diarized["overlaps"],
            "interruptions": diarized["interruptions"],
        }
    except AudioConversionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/collective")
async def collective_analysis(
    file: UploadFile = File(...),
    min_overlap_threshold: float = Form(0.2),
    cut_in_window: float = Form(1.0),
) -> dict[str, Any]:
    suffix = _safe_suffix(file.filename)
    if suffix and suffix not in _ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension '{suffix}'.")

    _require_positive("min_overlap_threshold", min_overlap_threshold)
    _require_positive("cut_in_window", cut_in_window)

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    transcript = {"full_text": "", "segments": []}
    speaker_utterances: list[dict[str, Any]] = []
    idea_map = {"main_ideas": []}
    analysis_error: str | None = None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "normalized.wav"
            convert_audio_bytes_to_wav16k_mono(raw_bytes, wav_path)

            diarized = diarize_wav(wav_path, min_overlap_threshold, cut_in_window)

            try:
                transcript = transcribe_with_whisper(wav_path)
                speaker_utterances = align_speakers_to_transcript(
                    diarized["segments_obj"], transcript["segments"]
                )
                sentences = split_into_idea_sentences(transcript["full_text"])
                idea_map = attach_speakers_to_ideas(cluster_ideas(sentences), speaker_utterances)
            except Exception as exc:
                analysis_error = f"Transcription or idea clustering failed: {exc}"

        scores = compute_scores(
            overlaps=diarized["overlaps"],
            interruptions=diarized["interruptions"],
            segments=diarized["segments_obj"],
            idea_map=idea_map,
        )
        if analysis_error:
            scores["debug"]["analysis_error"] = analysis_error

        return {
            "num_speakers": diarized["num_speakers"],
            "speakers": diarized["speakers"],
            "segments": diarized["segments"],
            "overlaps": diarized["overlaps"],
            "interruptions": diarized["interruptions"],
            "transcript": transcript,
            "speaker_utterances": speaker_utterances,
            "idea_map": idea_map,
            "scores": scores,
        }

    except AudioConversionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected analysis failure: {exc}") from exc
