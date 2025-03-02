"""
Run a batch job start to finish.
"""

import argparse
import sys
from pathlib import Path

from .batch import Batch
from .providers import _add_provider_args, _get_provider


def get_parser(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_file",
        type=str,
        nargs="?" if "--resume" in argv else 1,
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

    parser.add_argument(
        "--dry-run",
        "-n",
        help="Show what would happen but do not actually do it.",
        default=False,
        action="store_true",
    )

    _add_provider_args(parser)

    return parser


def main(args=None):
    args = get_parser(args or sys.argv).parse_args(args)
    provider = _get_provider(args)

    if args.resume:
        if args.dry_run:
            print(f"Would wait until {args.resume} is complete.")
            # For dry run, we still want to create a batch object and call wait with dry_run=True
            # This allows tests to verify the behavior without making API calls
            with Batch(output_file=args.output_file, error_file=args.error_file) as batch:
                batch.provider = provider
                batch.batch_id = args.resume
                # Wait for completion with dry_run=True
                batch.wait(
                    callback=lambda b: print(f"Status of {args.resume}: {b.status}"), dry_run=True
                )
            return args.resume

        # Create batch object for resuming
        with Batch(output_file=args.output_file, error_file=args.error_file) as batch:
            batch.provider = provider
            batch.batch_id = args.resume
            # Wait for completion
            batch.wait(callback=lambda b: print(f"Status of {args.resume}: {b.status}"))
        return args.resume

    if not args.input_file or not args.input_file[0]:
        print("Please specify an input file, or '-' to read from stdin.")
        return None

    # Get input
    input_file = sys.stdin.buffer.read() if args.input_file[0] == "-" else args.input_file[0]

    if args.dry_run:
        num_requests = (
            len(input_file.decode().splitlines())
            if isinstance(input_file, bytes)
            else sum(1 for _ in open(input_file))
        )
        print(
            f"Would start batch with {num_requests} requests then "
            + ("exit." if args.create else "wait for it to finish.")
        )

        # For dry run, we still want to create a batch object and call submit/wait with dry_run=True
        # This allows tests to verify the behavior without making API calls
        with Batch(
            submission_input_file=input_file,
            output_file=args.output_file,
            error_file=args.error_file,
        ) as batch:
            batch.provider = provider
            batch_id = batch.submit(dry_run=True)

            if args.create:
                return batch_id

            # Wait for completion with dry_run=True
            batch.wait(callback=lambda b: print(f"Status of {batch_id}: {b.status}"), dry_run=True)
        return batch_id

    # Create and submit batch
    with Batch(
        submission_input_file=input_file,
        output_file=args.output_file,
        error_file=args.error_file,
    ) as batch:
        batch.provider = provider
        batch_id = batch.submit(dry_run=args.dry_run)

        print(f"Created {batch_id}.")
        print("Processing may take anywhere from a few minutes to a few hours.")

        if args.create:
            print(f"This script now exits. Resume later with: --resume {batch_id}")
            return batch_id

        print("This script will now wait for batch to finish, then it will download the output.")
        print(f"You may Ctrl+C and resume later with: --resume {batch_id}")

        # Wait for completion
        batch.wait(
            callback=lambda b: print(f"Status of {batch_id}: {b.status}"), dry_run=args.dry_run
        )
    return batch_id


if __name__ == "__main__":
    main()
