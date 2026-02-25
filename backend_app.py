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

    token = os.getenv("HF_TOKEN")
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

        _PIPELINE = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
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
