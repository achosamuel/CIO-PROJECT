"""Shared audio helpers for conversion and WAV serialization."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile


PathLike = Union[str, Path]


class AudioConversionError(RuntimeError):
    """Raised when audio conversion fails."""


def convert_audio_to_wav16k_mono(input_path: PathLike, output_path: PathLike) -> Path:
    """Convert arbitrary input audio into mono, 16kHz WAV.

    Parameters
    ----------
    input_path:
        Source audio path (wav/mp3/m4a/etc.).
    output_path:
        Destination WAV path.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(16_000)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio.export(output_path, format="wav")
    except FileNotFoundError as exc:
        raise AudioConversionError(
            "ffmpeg/ffprobe is required by pydub but was not found. "
            "Install ffmpeg and ensure it is available on PATH."
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive conversion guard
        raise AudioConversionError(f"Could not convert audio file: {exc}") from exc

    return output_path


def convert_audio_bytes_to_wav16k_mono(input_bytes: bytes, output_path: PathLike) -> Path:
    """Convert in-memory audio bytes into mono 16kHz WAV."""
    output_path = Path(output_path)
    try:
        audio = AudioSegment.from_file(BytesIO(input_bytes))
        audio = audio.set_channels(1).set_frame_rate(16_000)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio.export(output_path, format="wav")
    except FileNotFoundError as exc:
        raise AudioConversionError(
            "ffmpeg/ffprobe is required by pydub but was not found. "
            "Install ffmpeg and ensure it is available on PATH."
        ) from exc
    except Exception as exc:  # pragma: no cover
        raise AudioConversionError(f"Could not convert uploaded audio: {exc}") from exc

    return output_path


def save_numpy_audio_to_wav(
    audio_array: np.ndarray,
    output_path: PathLike,
    sample_rate: int = 16_000,
) -> Path:
    """Save a numpy float/int audio array as 16-bit PCM WAV.

    Expects mono input. For multi-channel arrays, caller should down-mix first.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)

    # If floats are in [-1, 1], scale them.
    if np.issubdtype(audio_array.dtype, np.floating):
        clipped = np.clip(audio_array, -1.0, 1.0)
        pcm = (clipped * 32767).astype(np.int16)
    else:
        pcm = audio_array.astype(np.int16)

    wavfile.write(output_path, sample_rate, pcm)
    return output_path
