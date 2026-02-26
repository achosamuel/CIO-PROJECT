"""FastAPI backend for diarization + collective organization analysis.

Local run:
- PowerShell: $env:OPENAI_API_KEY="..." ; $env:HF_TOKEN="..."
- API server: uvicorn backend_app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import inspect
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from utils_audio import AudioConversionError, convert_audio_bytes_to_wav16k_mono

app = FastAPI(title="Speaker Counter + Interruption Detector")

# Helpful for local frontend-backend communication.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_PIPELINE = None
_OPENAI_CLIENT: OpenAI | None = None
_ALLOWED_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


@dataclass
class Segment:
    start: float
    end: float
    speaker: str


def get_openai_client() -> OpenAI:
    """Create/cached OpenAI client with API key from env."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is missing. Set it on the backend server and retry.",
        )

    _OPENAI_CLIENT = OpenAI(api_key=api_key, timeout=60.0)
    return _OPENAI_CLIENT


def get_diarization_pipeline() -> Any:
    """Create/cached pyannote pipeline with HF token."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    token = os.getenv("HF_TOKEN", "").strip()
    if not token or not token.startswith("hf_"):
        raise HTTPException(
            status_code=503,
            detail=(
                "Missing/invalid Hugging Face token. Set a valid HF_TOKEN (starts with 'hf_') "
                "and make sure you accepted model terms for pyannote/speaker-diarization-3.1."
            ),
        )

    try:
        from pyannote.audio import Pipeline

        from_pretrained_sig = inspect.signature(Pipeline.from_pretrained)
        auth_arg = "token" if "token" in from_pretrained_sig.parameters else "use_auth_token"

        _PIPELINE = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", **{auth_arg: token}
        )
        if _PIPELINE is None:
            raise RuntimeError("Pipeline.from_pretrained returned None")
        return _PIPELINE
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not load diarization model. Verify HF_TOKEN, accept model access "
                "permissions on Hugging Face, and ensure torch/pyannote are installed. "
                f"Underlying error: {exc}"
            ),
        ) from exc


def merge_overlap_events(overlap_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge overlap events that touch/intersect, combining speaker sets."""
    if not overlap_events:
        return []

    overlap_events = sorted(overlap_events, key=lambda x: (x["start"], x["end"]))
    merged: list[dict[str, Any]] = []

    for event in overlap_events:
        if not merged or event["start"] > merged[-1]["end"]:
            merged.append(
                {
                    "start": event["start"],
                    "end": event["end"],
                    "speakers": set(event["speakers"]),
                }
            )
        else:
            merged[-1]["end"] = max(merged[-1]["end"], event["end"])
            merged[-1]["speakers"].update(event["speakers"])

    finalized = []
    for event in merged:
        speakers_sorted = sorted(event["speakers"])
        finalized.append(
            {
                "start": round(event["start"], 3),
                "end": round(event["end"], 3),
                "speakers": speakers_sorted,
                "duration": round(event["end"] - event["start"], 3),
            }
        )
    return finalized


def detect_overlaps(segments: list[Segment], min_overlap: float = 0.2) -> list[dict[str, Any]]:
    """Find overlap windows across speaker segments."""
    events: list[dict[str, Any]] = []

    for i in range(len(segments)):
        a = segments[i]
        for j in range(i + 1, len(segments)):
            b = segments[j]
            if a.speaker == b.speaker:
                continue
            start = max(a.start, b.start)
            end = min(a.end, b.end)
            if end > start and (end - start) >= min_overlap:
                events.append(
                    {
                        "start": start,
                        "end": end,
                        "speakers": [a.speaker, b.speaker],
                    }
                )

    return merge_overlap_events(events)


def infer_interruptions(
    segments: list[Segment],
    overlaps: list[dict[str, Any]],
    cut_in_window: float = 1.0,
) -> list[dict[str, Any]]:
    """Infer interruptions from overlap windows."""
    interruptions: list[dict[str, Any]] = []

    for overlap in overlaps:
        o_start = overlap["start"]
        o_end = overlap["end"]

        interrupted = None
        interrupter = None

        for seg in segments:
            if seg.start <= o_start <= seg.end and seg.end >= o_start:
                was_already_talking = seg.start <= (o_start - 0.05)
                if was_already_talking:
                    interrupted = seg.speaker

        for seg in segments:
            starts_near_overlap = 0 <= (o_start - seg.start) <= cut_in_window
            active_during_overlap = seg.start <= o_start <= seg.end
            if starts_near_overlap and active_during_overlap and seg.speaker != interrupted:
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

    deduped = []
    seen = set()
    for item in interruptions:
        key = (item["start"], item["end"], item["interrupter"], item["interrupted"])
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped


def run_diarization(
    raw_bytes: bytes,
    filename: str,
    min_overlap_threshold: float,
    cut_in_window: float,
) -> dict[str, Any]:
    """Normalize audio, diarize, then compute overlap/interruptions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / (filename or "input_audio")
        wav_path = Path(tmpdir) / "normalized.wav"
        input_path.write_bytes(raw_bytes)

        try:
            convert_audio_bytes_to_wav16k_mono(raw_bytes, wav_path)
        except AudioConversionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        pipeline = get_diarization_pipeline()
        try:
            diarization = pipeline(str(wav_path))
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Diarization model is unavailable. Confirm HF_TOKEN has access to "
                    "pyannote/speaker-diarization-3.1 and that model terms are accepted. "
                    f"Underlying error: {exc}"
                ),
            ) from exc

        segments: list[Segment] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                Segment(
                    start=float(turn.start),
                    end=float(turn.end),
                    speaker=str(speaker),
                )
            )

    segments.sort(key=lambda s: (s.start, s.end))
    speakers = sorted({s.speaker for s in segments})
    overlaps = detect_overlaps(segments, min_overlap=min_overlap_threshold)
    interruptions = infer_interruptions(segments, overlaps, cut_in_window=cut_in_window)
    return {
        "segments_obj": segments,
        "num_speakers": len(speakers),
        "speakers": speakers,
        "segments": [
            {"start": round(s.start, 3), "end": round(s.end, 3), "speaker": s.speaker} for s in segments
        ],
        "overlaps": overlaps,
        "interruptions": interruptions,
    }


def fallback_transcript_segments(full_text: str, duration_seconds: float) -> list[dict[str, Any]]:
    """Create approximate timestamped text segments when timestamps are unavailable."""
    chunks = [c.strip() for c in re.split(r"(?<=[.!?])\s+", full_text) if c.strip()]
    if not chunks:
        return []
    if duration_seconds <= 0:
        duration_seconds = float(len(chunks))

    step = duration_seconds / len(chunks)
    out = []
    for i, text in enumerate(chunks):
        start = i * step
        end = duration_seconds if i == len(chunks) - 1 else (i + 1) * step
        out.append({"start": round(start, 3), "end": round(end, 3), "text": text})
    return out


def transcribe_audio_with_openai(wav_path: Path, duration_seconds: float) -> dict[str, Any]:
    """Transcribe audio in English with timestamps when possible."""
    client = get_openai_client()
    with wav_path.open("rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
            language="en",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    text = getattr(transcription, "text", "") or ""
    transcript_segments: list[dict[str, Any]] = []
    for seg in getattr(transcription, "segments", []) or []:
        transcript_segments.append(
            {
                "start": round(float(getattr(seg, "start", 0.0) or 0.0), 3),
                "end": round(float(getattr(seg, "end", 0.0) or 0.0), 3),
                "text": str(getattr(seg, "text", "") or "").strip(),
            }
        )

    if not transcript_segments and text.strip():
        transcript_segments = fallback_transcript_segments(text, duration_seconds)

    return {
        "full_text": text,
        "segments": [s for s in transcript_segments if s["text"]],
    }


def align_speakers_to_transcript(
    diarization_segments: list[Segment], transcript_segments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Assign each transcript segment to speaker with maximum temporal overlap."""
    utterances: list[dict[str, Any]] = []
    for tseg in transcript_segments:
        t_start = float(tseg.get("start", 0.0))
        t_end = float(tseg.get("end", t_start))
        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for dseg in diarization_segments:
            overlap = max(0.0, min(t_end, dseg.end) - max(t_start, dseg.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dseg.speaker

        utterances.append(
            {
                "speaker": best_speaker,
                "start": round(t_start, 3),
                "end": round(t_end, 3),
                "text": str(tseg.get("text", "")).strip(),
            }
        )

    return [u for u in utterances if u["text"]]


def extract_idea_map(full_text: str, speaker_utterances: list[dict[str, Any]]) -> dict[str, Any]:
    """Use OpenAI Responses API with strict JSON schema to build idea map."""
    if not full_text.strip():
        return {"main_ideas": []}

    client = get_openai_client()

    schema = {
        "name": "idea_map",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "main_ideas": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "summary": {"type": "string"},
                            "sub_ideas": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "text": {"type": "string"},
                                        "speakers": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "evidence": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": ["text", "speakers", "evidence"],
                                },
                            },
                        },
                        "required": ["id", "title", "summary", "sub_ideas"],
                    },
                }
            },
            "required": ["main_ideas"],
        },
    }

    prompt = (
        "You are an English brainstorming analysis assistant. "
        "Extract atomic ideas from this transcript, merge duplicates, and organize into a hierarchy "
        "of main ideas and sub-ideas. Keep titles concise. Keep summaries to 1-2 sentences. "
        "Use short evidence quotes from the transcript (not long passages). "
        "Return JSON only matching the schema."
    )

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Transcript:\n"
                            f"{full_text}\n\n"
                            "Speaker utterances (JSON):\n"
                            f"{speaker_utterances}"
                        ),
                    }
                ],
            },
        ],
        text={"format": {"type": "json_schema", "name": schema["name"], "strict": True, "schema": schema["schema"]}},
    )

    parsed = getattr(response, "output_parsed", None)
    if isinstance(parsed, dict) and "main_ideas" in parsed:
        return parsed

    # Fallback parse from text body if SDK doesn't populate output_parsed.
    text = response.output_text or ""
    if not text.strip():
        return {"main_ideas": []}
    import json

    payload = json.loads(text)
    if isinstance(payload, dict) and "main_ideas" in payload:
        return payload
    return {"main_ideas": []}


def gini(values: list[float]) -> float:
    """Compute Gini coefficient for non-negative values."""
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
    audio_duration_minutes = audio_duration / 60.0
    interruptions_per_min = len(interruptions) / (audio_duration_minutes + 1e-6)

    independence = 100.0 - (overlap_ratio * 80.0) - min(30.0, interruptions_per_min * 10.0)
    independence = max(0.0, min(100.0, independence))

    gini_speaking_time = gini(list(speaking_times.values()))
    participation_balance = max(0.0, min(100.0, (1.0 - gini_speaking_time) * 100.0))

    num_main_ideas = len(idea_map.get("main_ideas", []))
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
            "gini_speaking_time": round(gini_speaking_time, 4),
            "num_main_ideas": num_main_ideas,
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    min_overlap_threshold: float = Form(0.2),
    cut_in_window: float = Form(1.0),
) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix and suffix not in _ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{suffix}'. Use one of: {sorted(_ALLOWED_EXTS)}",
        )

    if min_overlap_threshold < 0:
        raise HTTPException(status_code=400, detail="min_overlap_threshold must be >= 0")
    if cut_in_window < 0:
        raise HTTPException(status_code=400, detail="cut_in_window must be >= 0")

    try:
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        diarized = run_diarization(
            raw_bytes=raw_bytes,
            filename=file.filename or "input_audio",
            min_overlap_threshold=min_overlap_threshold,
            cut_in_window=cut_in_window,
        )

        return {
            "num_speakers": diarized["num_speakers"],
            "speakers": diarized["speakers"],
            "segments": diarized["segments"],
            "overlaps": diarized["overlaps"],
            "interruptions": diarized["interruptions"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected analysis failure: {exc}",
        ) from exc


@app.post("/collective")
async def collective_analysis(
    file: UploadFile = File(...),
    min_overlap_threshold: float = Form(0.2),
    cut_in_window: float = Form(1.0),
) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix and suffix not in _ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{suffix}'. Use one of: {sorted(_ALLOWED_EXTS)}",
        )

    if min_overlap_threshold < 0:
        raise HTTPException(status_code=400, detail="min_overlap_threshold must be >= 0")
    if cut_in_window < 0:
        raise HTTPException(status_code=400, detail="cut_in_window must be >= 0")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    diarized = run_diarization(
        raw_bytes=raw_bytes,
        filename=file.filename or "input_audio",
        min_overlap_threshold=min_overlap_threshold,
        cut_in_window=cut_in_window,
    )

    transcript = {"full_text": "", "segments": []}
    speaker_utterances: list[dict[str, Any]] = []
    idea_map = {"main_ideas": []}
    analysis_error: str | None = None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "normalized.wav"
            convert_audio_bytes_to_wav16k_mono(raw_bytes, wav_path)
            audio_duration = max((s.end for s in diarized["segments_obj"]), default=0.0)
            transcript = transcribe_audio_with_openai(wav_path, audio_duration)

        speaker_utterances = align_speakers_to_transcript(diarized["segments_obj"], transcript["segments"])
        idea_map = extract_idea_map(transcript.get("full_text", ""), speaker_utterances)
    except HTTPException:
        raise
    except Exception as exc:
        analysis_error = f"Transcription or idea extraction failed: {exc}"

    scores = compute_scores(
        overlaps=diarized["overlaps"],
        interruptions=diarized["interruptions"],
        segments=diarized["segments_obj"],
        idea_map=idea_map,
    )
    if analysis_error:
        scores.setdefault("debug", {})["analysis_error"] = analysis_error

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
