"""FastAPI backend for speaker counting and interruption detection."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

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
_ALLOWED_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
TOKEN = "hf_iRzVdteghGHWYljbOQYAmPOJaYJM"


@dataclass
class Segment:
    start: float
    end: float
    speaker: str


def get_diarization_pipeline() -> Any:
    """Create/cached pyannote pipeline with HF token."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    token = os.getenv("HF_TOKEN", TOKEN)
    if not token:
        raise HTTPException(
            status_code=503,
            detail=(
                "Missing Hugging Face token. Set HF_TOKEN and make sure you accepted "
                "model terms for pyannote/speaker-diarization-3.1."
            ),
        )

    try:
        from pyannote.audio import Pipeline

        _PIPELINE = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
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
    """Infer interruptions from overlap windows.

    Heuristic:
    - interrupted speaker: active before overlap start and continuing into overlap.
    - interrupter: starts near overlap start (<= cut_in_window) and is active in overlap.
    """
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

    # Deduplicate if overlapping events produced identical interruption records.
    deduped = []
    seen = set()
    for item in interruptions:
        key = (item["start"], item["end"], item["interrupter"], item["interrupted"])
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped


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

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / (file.filename or "input_audio")
            wav_path = Path(tmpdir) / "normalized.wav"
            input_path.write_bytes(raw_bytes)

            try:
                convert_audio_bytes_to_wav16k_mono(raw_bytes, wav_path)
            except AudioConversionError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            pipeline = get_diarization_pipeline()
            diarization = pipeline(str(wav_path))

            segments: list[Segment] = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(
                    Segment(
                        start=float(turn.start),
                        end=float(turn.end),
                        speaker=str(speaker),
                    )
                )

        if not segments:
            return {
                "num_speakers": 0,
                "speakers": [],
                "segments": [],
                "overlaps": [],
                "interruptions": [],
            }

        segments.sort(key=lambda s: (s.start, s.end))
        speakers = sorted({s.speaker for s in segments})
        overlaps = detect_overlaps(segments, min_overlap=min_overlap_threshold)
        interruptions = infer_interruptions(segments, overlaps, cut_in_window=cut_in_window)

        return {
            "num_speakers": len(speakers),
            "speakers": speakers,
            "segments": [
                {"start": round(s.start, 3), "end": round(s.end, 3), "speaker": s.speaker}
                for s in segments
            ],
            "overlaps": overlaps,
            "interruptions": interruptions,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected analysis failure: {exc}",
        ) from exc
