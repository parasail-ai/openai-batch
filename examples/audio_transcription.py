# To run this example:
# export PARASAIL_API_KEY="psk-XYZ"
# python audio_transcription.py --audio-directory audio_files

import argparse
from pathlib import Path

from openai_batch import Batch, data_url

p = argparse.ArgumentParser()
p.add_argument(
    "--audio-directory", help="Directory of audio files", default="audio_files", type=Path
)
p.add_argument("--dry-run", help="Test without making actual API calls", action="store_true")
args = p.parse_args()
audio_dir = args.audio_directory.resolve()

# Create a batch that transcribes audio files
with Batch() as batch:
    audio_files = [
        p
        for p in Path(audio_dir).iterdir()
        if p.suffix.lower() in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
    ]

    print(f"Found {len(audio_files)} audio files")

    for audio_file in audio_files:
        print(f"Adding {audio_file.name}...")

        # Convert audio file to base64 data URL (similar to images)
        audio_data_url = data_url(audio_file)

        batch.add_to_batch(
            model="openai/whisper-large-v3",
            file=audio_data_url,
            response_format="text",
        )

    # Submit, wait for completion, and download results
    result, output_path, error_path = batch.submit_wait_download(dry_run=args.dry_run)
    print(f"Batch completed with status {result.status} and stored in {output_path}")
