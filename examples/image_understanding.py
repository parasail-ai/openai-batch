# To run this example:
# export PARASAIL_API_KEY="psk-XYZ"
# python image_understanding.py --image-directory images

import argparse
from pathlib import Path

from openai_batch import Batch, data_url

p = argparse.ArgumentParser()
p.add_argument("--image-directory", help="Directory of images", default="images", type=Path)
args = p.parse_args()
image_dir = args.image_directory.resolve()

# Create a batch that analyzes images
with Batch() as batch:
    images = (p for p in Path(image_dir).iterdir() if p.suffix.lower() in {".jpg", ".png", ".webp"})

    for image in images:
        batch.add_to_batch(
            model="Qwen/Qwen3-VL-8B-Instruct",
            max_completion_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url(image)}},
                        {"type": "text", "text": "What is in the image?"},
                    ],
                }
            ],
        )

    # Submit, wait for completion, and download results
    result, output_path, error_path = batch.submit_wait_download()
    print(f"Batch completed with status {result.status} and stored in {output_path}")
