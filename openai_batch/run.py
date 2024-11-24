"""
Run a batch job start to finish.
"""

import argparse
import sys
from pathlib import Path

from openai import OpenAI

from . import wait
from .providers import _add_provider_args, _get_provider


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_file",
        type=str,
        help="Batch input file in .jsonl format. '-' will read from stdin.",
    )

    parser.add_argument(
        "--output-file",
        "-o",
        dest="output_file",
        type=str,
        help="Output filename. '-' will write to stdout.",
    )

    parser.add_argument(
        "--error-file",
        "-e",
        dest="error_file",
        type=str,
        help="Error filename. '-' will write to stderr.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        help="Resume an existing batch (specify batch ID)",
    )

    parser.add_argument(
        "--create",
        "-c",
        help="Only create the batch, do not wait for it to finish.",
        default=False,
        action="store_true",
    )

    _add_provider_args(parser)

    return parser


def write(client, file_id, fname, dash_stream):
    contents = client.files.content(file_id).content

    if fname == "-":
        dash_stream.write(contents)
    else:
        print(f"Writing {fname}")
        Path(fname).write_bytes(contents)


def main(args=None):
    args = get_parser().parse_args(args)
    provider = _get_provider(args)
    client = OpenAI(base_url=provider.base_url, api_key=provider.api_key)

    batch_id = args.resume

    if not batch_id:
        if args.input_file == "-":
            input_file = sys.stdin.buffer.read()
        else:
            input_file = open(args.input_file, "rb")

        # Upload input file
        input_file = client.files.create(file=input_file, purpose="batch")

        # Create batch
        batch = client.batches.create(
            input_file_id=input_file.id,
            completion_window="24h",
            endpoint="/v1/chat/completions",
        )

        batch_id = batch.id

        print(f"Created {batch_id}.")
        print("Processing may take anywhere from a few minutes to a few hours.")
        if args.create:
            print(f"This script now exits. Resume later with: --resume {batch_id}")
            return batch_id

        print("This script will now wait for batch to finish, then it will download the output.")
        print(f"You may Ctrl+C and resume later with: --resume {batch_id}")

    # Wait for batch to complete
    batch = wait(client, batch_id, callback=lambda b: print(f"Status of {b.id}: {b.status}"))

    # Download output file
    if batch.output_file_id:
        write(
            client,
            batch.output_file_id,
            args.output_file or f"{batch_id}-output.jsonl",
            sys.stdout,
        )

    # Download error file
    if batch.error_file_id:
        write(
            client,
            batch.error_file_id,
            args.error_file or f"{batch_id}-errors.jsonl",
            sys.stderr,
        )

    return batch_id


if __name__ == "__main__":
    main()