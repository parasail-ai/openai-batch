"""
Construct an input file from prompts
"""

import argparse

from .batch import Batch
from .providers import _add_provider_arg


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input",
        nargs="?",
        default="-",
        type=argparse.FileType("r"),
        help="Filename or '-' for stdin. One prompt per line.",
    )

    parser.add_argument(
        "output",
        nargs="?",
        default="-",
        type=argparse.FileType("w", encoding="utf-8"),
        help="Filename or '-' for stdout. This will be the batch input .jsonl file.",
    )

    parser.add_argument(
        "--model",
        default="gpt-4",
        help="Which model to target.",
    )

    parser.add_argument(
        "--embedding",
        "-e",
        help="Whether this is an embedding model",
        default=False,
        action="store_true",
    )

    # Need to know batch input file size limits, this can vary by provider.
    _add_provider_arg(parser)

    return parser


def main(args=None):
    args = get_parser().parse_args(args)

    with Batch(output_file=args.output) as batch:
        for prompt in args.input:
            if args.embedding:
                batch.add_to_batch(model=args.model, input=prompt.rstrip())
            else:
                batch.add_to_batch(
                    model=args.model, messages=[{"role": "user", "content": prompt.rstrip()}]
                )


if __name__ == "__main__":
    main()
