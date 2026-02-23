#!/usr/bin/env python3
"""
Simple Batch Audio Transcription Example

This script demonstrates batch transcription of multiple audio files using the OpenAI Batch SDK.

Prerequisites:
- Export OpenAI API key: export OPENAI_API_KEY="your-key-here"
- Or export Parasail API key: export PARASAIL_API_KEY="your-key-here"

Usage:
    # Transcribe audio files
    python audio_transcription.py --audio-directory audio_files

    # Use different model
    python audio_transcription.py --audio-directory audio_files --model openai/whisper-tiny

    # Test without API calls
    python audio_transcription.py --audio-directory audio_files --dry-run
"""

import argparse
from pathlib import Path
from openai_batch import Batch, data_url


def main():
    parser = argparse.ArgumentParser(description="Batch transcribe audio files")
    parser.add_argument(
        "--audio-directory",
        help="Directory containing audio files",
        default="audio_files",
        type=Path,
    )
    parser.add_argument("--model", help="Whisper model to use", default="openai/whisper-large-v3")
    parser.add_argument(
        "--dry-run", help="Test without making actual API calls", action="store_true"
    )
    args = parser.parse_args()

    # Resolve directory path
    audio_dir = args.audio_directory.resolve()

    # Find audio files
    audio_extensions = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".webm"}
    audio_files = [
        p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() in audio_extensions
    ]

    if not audio_files:
        print(f"✗ No audio files found in {audio_dir}")
        return

    print(f"{'=' * 50}")
    print(f"Batch Audio Transcription")
    print(f"{'=' * 50}")
    print(f"Model: {args.model}")
    print(f"Audio files: {len(audio_files)}")
    print(f"{'=' * 50}")

    # Create batch
    with Batch() as batch:
        # Add each audio file for transcription
        for i, audio_file in enumerate(audio_files, 1):
            print(f"[{i}/{len(audio_files)}] Processing: {audio_file.name}")

            # Convert audio file to base64 data URL
            audio_data_url = data_url(audio_file)

            batch.add_to_batch(
                model=args.model,
                file=audio_data_url,
                response_format="text",
            )

        print(f"\n✓ Batch configured with {batch.n_requests} audio files")

        # Submit, wait for completion, and download results
        result, output_path, error_path = batch.submit_wait_download(
            dry_run=args.dry_run,
        )

        print(f"\n{'=' * 50}")
        print(f"Batch Status: {result.status}")
        print(f"Output saved to: {output_path}")
        print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
