#!/usr/bin/env python3
"""
Batch Audio Transcription Example

This script demonstrates how to batch transcribe multiple audio files using the OpenAI Batch SDK.

Prerequisites:
- Install ffmpeg (for system audio processing if needed): sudo apt-get install ffmpeg
- Export OpenAI API key: export OPENAI_API_KEY="your-key-here"
- Or export Parasail API key: export PARASAIL_API_KEY="your-key-here"

Usage:
    # Transcribe audio files in a directory
    python batch_transcribe.py --audio-directory ./audio_files

    # Use different whisper model
    python batch_transcribe.py --audio-directory ./audio_files --model openai/whisper-tiny

    # Test without API calls
    python batch_transcribe.py --audio-directory ./audio_files --dry-run

    # Process multiple directories
    python batch_transcribe.py --audio-directory ./audio_files --output-file my_transcriptions.jsonl
"""

import argparse
import json
from pathlib import Path
from openai_batch import Batch, data_url
from openai.types.batch import Batch as OpenAIBatch


def batch_transcribe(
    audio_directory: Path,
    model: str = "openai/whisper-large-v3",
    output_file: str = "transcriptions.jsonl",
    error_file: str = "transcription_errors.jsonl",
    dry_run: bool = False,
) -> tuple[OpenAIBatch, str, str]:
    """
    Batch transcribe audio files using OpenAI Batch SDK.

    Args:
        audio_directory: Directory containing audio files
        model: Whisper model to use
        output_file: Output file for successful transcriptions
        error_file: Error file for failed transcriptions
        dry_run: If True, skip actual API calls

    Returns:
        tuple: (batch_result, output_path, error_path)
    """

    # Find audio files (support multiple formats)
    audio_extensions = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".webm"}
    audio_files = [
        p for p in audio_directory.iterdir() if p.is_file() and p.suffix.lower() in audio_extensions
    ]

    audio_files.sort()

    if not audio_files:
        print(f"✗ No audio files found in {audio_directory}")
        return None, None, None

    print(f"{'=' * 60}")
    print(f"Batch Audio Transcription")
    print(f"{'=' * 60}")
    print(f"Model: {model}")
    print(f"Audio files: {len(audio_files)}")
    print(f"Output: {output_file}")
    print(f"Errors: {error_file}")

    # Create batch
    with Batch(output_file=output_file, error_file=error_file) as batch:
        # Add each audio file for transcription
        for i, audio_file in enumerate(audio_files, 1):
            print(f"[{i}/{len(audio_files)}] Processing: {audio_file.name}")

            # Convert audio file to base64 data URL
            audio_data_url = data_url(audio_file)

            batch.add_to_batch(
                model=model,
                file=audio_data_url,
                response_format="text",
            )

        print(f"\n✓ Batch configured with {batch.n_requests} audio files")

        # Submit, wait for completion, and download results
        batch_result, output_path, error_path = batch.submit_wait_download(
            dry_run=dry_run,
        )

        print(f"\n{'=' * 60}")
        print(f"Batch Status: {batch_result.status}")
        print(f"{'=' * 60}")

    return batch_result, output_path, error_path


def process_transcription_results(output_path: str, error_path: str):
    """Process and display transcription results."""

    if not output_path or not Path(output_path).exists():
        print("\n✗ No transcription results found")
        return

    print(f"\n{'=' * 60}")
    print("Processing Results")
    print(f"{'=' * 60}")

    successful_transcriptions = []
    failed_transcriptions = []

    # Read successful results
    with open(output_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                request_data = json.loads(line)
                response_data = request_data.get("response", {})
                content = response_data.get("content", [])

                # Extract transcript text from response
                transcript = ""
                for choice in content:
                    if choice.get("text"):
                        transcript += choice["text"]

                successful_transcriptions.append(
                    {"line": line_num, "transcript": transcript.strip()}
                )
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                continue

    # Read failed results
    if error_path and Path(error_path).exists():
        with open(error_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    failed_transcriptions.append({"error": line.strip()})

    # Display statistics
    print(f"\n✓ Successful transcriptions: {len(successful_transcriptions)}")
    print(f"✗ Failed transcriptions: {len(failed_transcriptions)}")

    # Display sample transcripts
    print(f"\n{'=' * 60}")
    print("Sample Transcripts (first 5):")
    print(f"{'=' * 60}")

    for i, item in enumerate(successful_transcriptions[:5], 1):
        print(f"\n[{i}] Line {item['line']}:")
        # Truncate long transcripts for display
        display_text = (
            item["transcript"][:300] + "..."
            if len(item["transcript"]) > 300
            else item["transcript"]
        )
        print(f"    {display_text}")

    # Create individual transcript files
    transcript_dir = Path("transcripts")
    transcript_dir.mkdir(exist_ok=True)

    for item in successful_transcriptions:
        transcript_path = transcript_dir / f"transcript_line_{item['line']}.txt"
        transcript_path.write_text(item["transcript"])

    print(f"\n{'=' * 60}")
    print(f"Individual transcripts saved to: {transcript_dir}/")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch transcribe multiple audio files using OpenAI Batch SDK"
    )

    parser.add_argument(
        "--audio-directory",
        help="Directory containing audio files",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model",
        help="Whisper model to use",
        default="openai/whisper-large-v3",
        choices=[
            "openai/whisper-tiny",
            "openai/whisper-base",
            "openai/whisper-small",
            "openai/whisper-medium",
            "openai/whisper-large-v2",
            "openai/whisper-large-v3",
        ],
    )
    parser.add_argument(
        "--output-file",
        help="Output file for batch transcription results",
        default="transcriptions.jsonl",
    )
    parser.add_argument(
        "--error-file",
        help="Error file for batch transcription results",
        default="transcription_errors.jsonl",
    )
    parser.add_argument(
        "--dry-run", help="Test without making actual API calls", action="store_true"
    )
    parser.add_argument(
        "--provider",
        help="Batch provider (openai or parasail)",
        choices=["openai", "parasail"],
        default=None,
    )

    args = parser.parse_args()

    # Resolve directory path
    args.audio_directory = args.audio_directory.resolve()

    # Create provider if specified
    from openai_batch import get_provider_by_name

    if args.provider:
        provider = get_provider_by_name(args.provider)
    else:
        provider = None

    # Run batch transcription
    batch_result, output_path, error_path = batch_transcribe(
        audio_directory=args.audio_directory,
        model=args.model,
        output_file=args.output_file,
        error_file=args.error_file,
        dry_run=args.dry_run,
    )

    if not batch_result:
        return

    # Process results
    process_transcription_results(output_path, error_path)

    print(f"\n{'=' * 60}")
    print("Processing Complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
