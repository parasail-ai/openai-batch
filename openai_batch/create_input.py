"""
Construct an input file from prompts
"""

import argparse
import json
import typing

import openai
from openai.types.chat import ChatCompletionUserMessageParam
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming
from openai.types import EmbeddingCreateParams

from .providers import _add_provider_arg, _get_provider

AUTO = "auto"

KNOWN_EMBEDDING_MODELS = [
    # HuggingFace
    "intfloat/e5-mistral-7b-instruct",
    "GritLM/GritLM-7B",
    "GritLM/GritLM-8x7B",
    # OpenAI
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
] + list(typing.get_args(openai.types.EmbeddingModel))


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
        default=AUTO,
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


def create_request(custom_id, body):
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/embeddings" if "input" in body else "/v1/chat/completions",
        "body": body,
    }


def main(args=None):
    args = get_parser().parse_args(args)
    provider = _get_provider(args)

    if args.model == AUTO:
        if args.embedding:
            args.model = provider.default_embedding_model
        else:
            args.model = provider.default_chat_model

    embedding = args.embedding or args.model in KNOWN_EMBEDDING_MODELS
    n_bytes = 0

    for i, prompt in enumerate(args.input):
        if i > provider.batch_input_max_requests:
            # Todo: auto batch splitting
            raise ValueError(
                f"Exceeded max number of requests per batch ({provider.batch_input_max_requests})"
            )

        prompt = prompt.rstrip()

        if embedding:
            body = EmbeddingCreateParams(
                encoding_format="base64",
                model=args.model,
                input=prompt,
            )
        else:
            body = CompletionCreateParamsNonStreaming(
                model=args.model,
                max_completion_tokens=1024,
                messages=[ChatCompletionUserMessageParam(role="user", content=prompt)],
            )
        request = create_request(custom_id=f"line-{i+1}", body=body)

        line = json.dumps(request) + "\n"
        n_bytes += len(line.encode("utf-8"))

        if n_bytes > provider.batch_input_max_bytes:
            # Todo: auto batch splitting
            raise ValueError(
                f"Exceeded max batch input file size ({provider.batch_input_max_bytes // 1024 // 1024} MB)"
            )

        args.output.write(line)


if __name__ == "__main__":
    main()
