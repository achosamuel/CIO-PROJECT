"""CLI helper to record microphone audio and send to backend /analyze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
import sounddevice as sd

from utils_audio import save_numpy_audio_to_wav


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record audio and analyze speakers/interruptions.")
    parser.add_argument("--duration", type=int, default=10, help="Recording duration in seconds.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate (Hz).")
    parser.add_argument("--output", type=str, default="recording.wav", help="Output WAV filename.")
    parser.add_argument("--backend-url", type=str, default="http://localhost:8000/analyze")
    parser.add_argument("--min-overlap-threshold", type=float, default=0.2)
    parser.add_argument("--cut-in-window", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Recording {args.duration}s from microphone at {args.sample_rate} Hz...")
    recording = sd.rec(
        int(args.duration * args.sample_rate),
        samplerate=args.sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    output_path = Path(args.output)
    save_numpy_audio_to_wav(recording.squeeze(), output_path, sample_rate=args.sample_rate)
    print(f"Saved recording to {output_path.resolve()}")

    with open(output_path, "rb") as f:
        response = requests.post(
            args.backend_url,
            files={"file": (output_path.name, f, "audio/wav")},
            data={
                "min_overlap_threshold": args.min_overlap_threshold,
                "cut_in_window": args.cut_in_window,
            },
            timeout=600,
        )

    if response.status_code != 200:
        print(f"Backend returned error {response.status_code}: {response.text}")
        return

    payload = response.json()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
