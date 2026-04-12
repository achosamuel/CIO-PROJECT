"""
FastAPI backend for diarization + collective organization + idea map (FREE).

Pipeline:
- Normalize audio to WAV 16k mono (utils_audio.py)
- Speaker diarization (pyannote) using HF_TOKEN
- Transcription (offline) using faster-whisper
- Speaker name inference: real names extracted from transcript instead of SPEAKER_00
- Idea map via Groq-hosted Llama 3.3 70B (full transcript, no truncation)
  with TF-IDF clustering as offline fallback

Run (PowerShell):
  uvicorn backend_app:app --host 127.0.0.1 --port 8000 --reload
""
Run (PowerShell):
  uvicorn backend_app:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

import inspect
import json
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
        "https://id-preview--7b3df4b1-7a21-406a-8ddf-a1bfdc3c5a50.lovable.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ALLOWED_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

# Caches (loaded once, reused)
_PIPELINE: Any | None = None
_WHISPER_MODEL = None
_GROQ_CHAT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


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
# Speaker name inference
# -------------------------

# Common filler/non-name words to reject when scanning first-word fallback
_NON_NAME_WORDS = frozenset({
    "i", "the", "a", "an", "so", "well", "yeah", "yes", "no", "ok", "okay",
    "hi", "hey", "hello", "good", "great", "right", "sure", "thanks", "thank",
    "let", "just", "we", "it", "is", "my", "our", "its", "um", "uh", "like",
    "and", "but", "or", "if", "of", "to", "in", "on", "at", "by", "for",
})

# Patterns for self-introduction: "I'm John", "I am Sarah", "My name is Alex"
_INTRO_PATTERNS = [
    re.compile(r"\bI'?m\s+([A-Z][a-z]{1,20})\b"),
    re.compile(r"\bI\s+am\s+([A-Z][a-z]{1,20})\b"),
    re.compile(r"\bmy\s+name(?:'?s|\s+is)\s+([A-Z][a-z]{1,20})\b", re.IGNORECASE),
    re.compile(r"\bthis\s+is\s+([A-Z][a-z]{1,20})\b"),
    re.compile(r"\bcall\s+me\s+([A-Z][a-z]{1,20})\b", re.IGNORECASE),
]

# Patterns for being addressed: "John, what do you think?" or "Thanks, Sarah."
# Captures a name that appears at the very start or end of an utterance (vocative)
_VOCATIVE_PATTERN = re.compile(
    r"(?:^|[,\.!?]\s*)([A-Z][a-z]{1,20})\s*[,\.!?]|[,\.!?]\s*([A-Z][a-z]{1,20})\s*[,\.!?]?$"
)


def infer_speaker_names(
    utterances: list[dict[str, Any]],
) -> dict[str, str]:
    """
    Attempt to infer a real first name for each diarized speaker label.

    Strategy (in priority order):
    1. Self-introduction patterns ("I'm John", "My name is Sarah") in the
       speaker's own utterances.
    2. Vocative address patterns — another speaker says "Hey John," or
       "Thanks, Sarah" — the addressed name is attributed to the silent speaker
       who was most recently active when the address occurred.
    3. First capitalised word of the speaker's first utterance that is not a
       stop-word / filler, used only when confidence is high (len >= 3, title-case).
    4. Fall back to the original label if nothing is found.

    Returns a mapping {original_label: display_name}.
    """
    if not utterances:
        return {}

    # Build ordered list of unique speakers
    all_speakers: list[str] = []
    seen: set[str] = set()
    for u in utterances:
        spk = u.get("speaker", "UNKNOWN")
        if spk not in seen:
            seen.add(spk)
            all_speakers.append(spk)

    name_map: dict[str, str] = {spk: spk for spk in all_speakers}
    confidence: dict[str, int] = {spk: 0 for spk in all_speakers}  # higher = more certain

    # Pass 1 — self-introductions (highest confidence = 3)
    for u in utterances:
        spk = u.get("speaker", "UNKNOWN")
        text = u.get("text", "") or ""
        for pat in _INTRO_PATTERNS:
            m = pat.search(text)
            if m:
                candidate = m.group(1).strip()
                if len(candidate) >= 2 and confidence[spk] < 3:
                    name_map[spk] = candidate
                    confidence[spk] = 3
                break

    # Pass 2 — vocative address (confidence = 2)
    # When speaker A addresses "John,", the name likely belongs to another speaker
    # We attribute it to the last active speaker that is NOT speaker A.
    last_active: dict[str, float] = {spk: -1.0 for spk in all_speakers}
    for u in utterances:
        spk = u.get("speaker", "UNKNOWN")
        t_start = float(u.get("start", 0.0))
        text = u.get("text", "") or ""

        for m in _VOCATIVE_PATTERN.finditer(text):
            candidate = (m.group(1) or m.group(2) or "").strip()
            if not candidate or len(candidate) < 2:
                continue
            # Find the speaker whose label we haven't resolved yet and was recently active
            for other_spk in all_speakers:
                if other_spk == spk:
                    continue
                recently = (last_active[other_spk] >= 0) and (t_start - last_active[other_spk] < 30.0)
                if recently and confidence[other_spk] < 2:
                    name_map[other_spk] = candidate
                    confidence[other_spk] = 2
                    break

        last_active[spk] = t_start

    # Pass 3 — first capitalised word of first utterance (confidence = 1)
    first_seen: set[str] = set()
    for u in utterances:
        spk = u.get("speaker", "UNKNOWN")
        if spk in first_seen:
            continue
        first_seen.add(spk)
        if confidence[spk] > 0:
            continue
        text = (u.get("text", "") or "").strip()
        words = text.split()
        for word in words[:6]:  # only look at first 6 words
            clean = re.sub(r"[^A-Za-z]", "", word)
            if (
                len(clean) >= 3
                and clean[0].isupper()
                and clean.lower() not in _NON_NAME_WORDS
            ):
                name_map[spk] = clean
                confidence[spk] = 1
                break

    # Deduplicate: if two speakers got the same inferred name, keep only the
    # one with higher confidence; reset the other to its original label.
    used_names: dict[str, str] = {}  # name -> speaker with highest confidence
    for spk in all_speakers:
        name = name_map[spk]
        if name == spk:
            continue  # still original label, no conflict possible
        if name not in used_names:
            used_names[name] = spk
        else:
            existing_spk = used_names[name]
            if confidence[spk] >= confidence[existing_spk]:
                # Current speaker wins, reset existing
                name_map[existing_spk] = existing_spk
                used_names[name] = spk
            else:
                # Existing wins, reset current
                name_map[spk] = spk

    return name_map


def apply_speaker_names(
    name_map: dict[str, str],
    *,
    segments: list[dict[str, Any]] | None = None,
    utterances: list[dict[str, Any]] | None = None,
    overlaps: list[dict[str, Any]] | None = None,
    interruptions: list[dict[str, Any]] | None = None,
    idea_map: dict[str, Any] | None = None,
) -> None:
    """
    In-place rename all speaker labels across every output structure.
    Mutates the objects directly so callers don't need to reassemble them.
    """
    if not name_map:
        return

    def remap(label: str) -> str:
        return name_map.get(label, label)

    if segments:
        for seg in segments:
            seg["speaker"] = remap(seg.get("speaker", ""))

    if utterances:
        for u in utterances:
            u["speaker"] = remap(u.get("speaker", ""))

    if overlaps:
        for ov in overlaps:
            ov["speakers"] = [remap(s) for s in ov.get("speakers", [])]

    if interruptions:
        for it in interruptions:
            it["interrupter"] = remap(it.get("interrupter", ""))
            it["interrupted"] = remap(it.get("interrupted", ""))

    if idea_map:
        for idea in idea_map.get("main_ideas", []):
            for sub in idea.get("sub_ideas", []):
                sub["speakers"] = [remap(s) for s in sub.get("speakers", [])]


# -------------------------
# HF + pyannote diarization
# -------------------------

def get_diarization_pipeline() -> Any:
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

        sig = inspect.signature(Pipeline.from_pretrained)
        if "token" in sig.parameters:
            _PIPELINE = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
        elif "use_auth_token" in sig.parameters:
            _PIPELINE = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        else:
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
                    "timestamp": round(o_start, 3),  # Added this line
                }
            )

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

    model_size = os.getenv("WHISPER_MODEL_SIZE", "base").strip()
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
# Idea map (offline: TF-IDF + clustering) — fallback only
# -------------------------

def split_into_idea_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    parts = [p for p in parts if len(p) >= 10]
    return parts[:300]


def cluster_ideas(sentences: list[str]) -> dict[str, Any]:
    """
    Offline fallback: TF-IDF + agglomerative clustering.
    No torch / no transformers.
    """
    if not sentences:
        return {"main_ideas": []}

    vec = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    X = vec.fit_transform(sentences)

    dist_thresh = float(os.getenv("IDEA_CLUSTER_DISTANCE", "0.70"))
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=dist_thresh,
    )

    labels = clustering.fit_predict(X.toarray()).tolist()

    clusters: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(i)

    cluster_items = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)

    main_ideas = []
    for ci, (_lab, idxs) in enumerate(cluster_items, start=1):
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


# -------------------------
# Idea map via Llama 3.3 70B (full transcript, no truncation)
# -------------------------

def _build_transcript_payload(
    speaker_utterances: list[dict[str, Any]],
    full_text: str,
) -> str:
    """
    Build the transcript string sent to the LLM.

    Preference: speaker-labelled utterances (most informative for idea attribution).
    Fallback: raw full_text if utterances are empty.

    No length truncation — we send the full conversation so the model can
    distinguish meta-talk (greetings, acknowledgements) from real ideas.
    """
    lines: list[str] = []
    for u in speaker_utterances:
        speaker = str(u.get("speaker", "UNKNOWN")).strip() or "UNKNOWN"
        start = float(u.get("start", 0.0) or 0.0)
        end = float(u.get("end", start) or start)
        text = str(u.get("text", "")).strip()
        if text:
            lines.append(f"[{start:.1f}-{end:.1f}] {speaker}: {text}")

    return "\n".join(lines).strip() or full_text.strip()


def extract_ideas_with_llama(
    full_text: str,
    speaker_utterances: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Use Groq-hosted Llama 3.3 70B to extract a structured idea map from the
    COMPLETE transcript (no truncation).

    The system prompt explicitly instructs the model to:
    - Ignore greetings, small talk, filler, acknowledgements.
    - Only surface concrete proposals, decisions, risks, strategies, or opportunities.
    - Return strict JSON, nothing else.

    Falls back to offline TF-IDF clustering when GROQ_API_KEY is absent or
    the request fails.
    """
    text = (full_text or "").strip()
    if not text:
        return {"main_ideas": []}

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY")

    model = os.getenv("IDEA_MODEL_NAME", "llama-3.3-70b-versatile").strip()

    transcript_payload = _build_transcript_payload(
        speaker_utterances or [],
        full_text,
    )

    # ---------------------------------------------------------------
    # System prompt — strict JSON schema + noise-filtering instructions
    # ---------------------------------------------------------------
    system_prompt = (
        "You are a precise meeting analyst. Your only output is a valid JSON object — "
        "no markdown, no prose, no explanation, no code fences.\n\n"
        "You receive a meeting transcript with timestamped speaker turns. "
        "Your task: extract the real ideas discussed.\n\n"
        "STRICT RULES:\n"
        "1. A valid idea is a concrete proposal, suggestion, strategy, decision, challenge, "
        "risk, or opportunity that was actually discussed in depth.\n"
        "2. NEVER include: greetings (hi, hello, good morning), farewells, small talk, "
        "filler words (yeah, ok, right, sure), acknowledgements (thanks, got it), "
        "one-word or one-phrase turns with no substance, or meta-conversation about the meeting itself.\n"
        "3. Sub-ideas must be specific and actionable. Minimum 15 characters. No filler.\n"
        "4. Evidence quotes must be verbatim short excerpts from the transcript.\n"
        "5. 3 to 8 main ideas total. 1 to 6 sub_ideas each.\n"
        "6. Output ONLY the JSON object below — nothing before or after it.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "main_ideas": [\n'
        "    {\n"
        '      "id": "I1",\n'
        '      "title": "Short title (max 80 chars)",\n'
        '      "summary": "1-2 sentence summary of the idea",\n'
        '      "sub_ideas": [\n'
        "        {\n"
        '          "text": "Specific actionable point",\n'
        '          "speakers": ["SpeakerName"],\n'
        '          "evidence": ["short verbatim quote"]\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}"
    )

    user_prompt = (
        "Here is the full meeting transcript. "
        "Extract the real ideas following your instructions exactly.\n\n"
        f"TRANSCRIPT:\n{transcript_payload}"
        # No [:N] slice — full transcript is sent
    )

    try:
        import requests
    except Exception as exc:
        raise RuntimeError(f"requests import failed: {exc}") from exc

    resp = requests.post(
        _GROQ_CHAT_ENDPOINT,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0.1,          # lower temp → fewer hallucinations / filler ideas
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        },
        timeout=90,   # longer timeout for large transcripts
    )
    resp.raise_for_status()
    data = resp.json()

    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    ).strip()
    if not content:
        raise RuntimeError("LLM returned empty content")

    # Strip accidental markdown code fences just in case
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    parsed = json.loads(content)
    main_ideas = parsed.get("main_ideas", [])
    if not isinstance(main_ideas, list):
        raise RuntimeError("Invalid JSON schema: main_ideas must be a list")

    # Post-processing: validate + sanitise every item
    noise_patterns = [
        re.compile(r"^\s*(thanks?|thank you|okay|ok|yeah|yep|nope|right|got it|cool|sure)\s*[.!]*\s*$", re.I),
        re.compile(r"^\s*(hello|hi|good morning|good afternoon|good evening)\b", re.I),
        re.compile(r"^\s*(bye|goodbye|see you|talk later)\b", re.I),
    ]

    cleaned: list[dict[str, Any]] = []
    for i, idea in enumerate(main_ideas[:12], start=1):
        if not isinstance(idea, dict):
            continue
        title = str(idea.get("title", f"Idea {i}")).strip()[:160]
        summary = str(idea.get("summary", "")).strip()[:500]

        sub_ideas: list[dict[str, Any]] = []
        for sub in (idea.get("sub_ideas", []) or [])[:12]:
            if not isinstance(sub, dict):
                continue
            sub_text = str(sub.get("text", "")).strip()
            if not sub_text or len(sub_text) < 15:
                continue
            if any(p.match(sub_text) for p in noise_patterns):
                continue

            evidence = sub.get("evidence", [])
            speakers = sub.get("speakers", [])
            sub_ideas.append(
                {
                    "text": sub_text[:300],
                    "speakers": [str(s).strip()[:40] for s in (speakers if isinstance(speakers, list) else []) if str(s).strip()],
                    "evidence": [str(e).strip()[:180] for e in (evidence if isinstance(evidence, list) else []) if str(e).strip()][:3],
                }
            )

        # Skip ideas that have no substance at all
        if not sub_ideas and len(summary) < 20:
            continue

        cleaned.append(
            {
                "id": f"I{i}",
                "title": title or f"Idea {i}",
                "summary": summary,
                "sub_ideas": sub_ideas,
            }
        )

    return {"main_ideas": cleaned}


def attach_speakers_to_ideas(idea_map: dict[str, Any], utterances: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Attribute speakers to sub-ideas by substring match of the first 30 chars
    of the sub-idea text against utterance texts.
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
            out[pkg] = getattr(mod, "__version__", "no __version__")
        except Exception as e:
            out[pkg] = f"IMPORT_ERROR: {type(e).__name__}: {e}"

    for pkg in ("torch", "huggingface_hub", "pyannote", "pyannote.audio", "faster_whisper", "sklearn", "numpy"):
        add(pkg)
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

    transcript: dict[str, Any] = {"full_text": "", "segments": []}
    speaker_utterances: list[dict[str, Any]] = []
    idea_map: dict[str, Any] = {"main_ideas": []}
    analysis_error: str | None = None
    speaker_name_map: dict[str, str] = {}

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

                # ── Infer real speaker names before passing to LLM ──────────
                # This runs on the raw SPEAKER_XX labels so the LLM already
                # receives human-readable names in the transcript payload.
                speaker_name_map = infer_speaker_names(speaker_utterances)
                apply_speaker_names(
                    speaker_name_map,
                    utterances=speaker_utterances,
                )

                try:
                    # Full transcript, no truncation
                    idea_map = extract_ideas_with_llama(
                        transcript["full_text"],
                        speaker_utterances=speaker_utterances,
                    )
                except Exception as llm_exc:
                    analysis_error = f"LLM idea extraction failed (using offline fallback): {llm_exc}"
                    sentences = split_into_idea_sentences(transcript["full_text"])
                    idea_map = cluster_ideas(sentences)

                idea_map = attach_speakers_to_ideas(idea_map, speaker_utterances)

            except Exception as exc:
                analysis_error = f"Transcription or idea clustering failed: {exc}"

        # Apply speaker names to diarization outputs too
        apply_speaker_names(
            speaker_name_map,
            segments=diarized["segments"],
            overlaps=diarized["overlaps"],
            interruptions=diarized["interruptions"],
            idea_map=idea_map,
        )

        # Remap the top-level speakers list
        renamed_speakers = sorted({
            speaker_name_map.get(s, s) for s in diarized["speakers"]
        })

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
            "speakers": renamed_speakers,
            "speaker_name_map": speaker_name_map,   # expose mapping for debugging
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
