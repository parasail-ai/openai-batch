import argparse
import os
from argparse import Namespace
import dataclasses
from dataclasses import dataclass


@dataclass
class Provider:
    name: str = None
    display_name: str = None
    base_url: str = None

    api_key: str = None
    api_key_env_var: str = None

    batch_input_max_requests: int = 50_000
    batch_input_max_bytes: int = 100 * 1024 * 1024


openai_provider = Provider(
    name="openai",
    display_name="OpenAI",
    base_url="https://api.openai.com/v1",
    api_key_env_var="OPENAI_API_KEY",
)


parasail_provider = Provider(
    name="parasail",
    display_name="Parasail",
    base_url="https://api.saas.parasail.io/v1",
    api_key_env_var="PARASAIL_API_KEY",
)

all_providers = [openai_provider, parasail_provider]


def _add_provider_arg(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        choices=[p.name for p in all_providers],
        default=None,
        help=f"Batch provider ({','.join(p.display_name for p in all_providers)})",
    )


def _add_provider_args(parser: argparse.ArgumentParser):
    _add_provider_arg(parser)

    parser.add_argument(
        "--base-url",
        type=str,
        help="The API base URL to use instead of specifying a known provider.",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        help=f"API Key to use. If omitted will attempt to fetch value from appropriate environment variable: "
        f"{','.join(p.api_key_env_var for p in all_providers)}",
    )


def _get_provider(args: Namespace) -> Provider:
    provider = Provider()

    if args.provider:
        for p in all_providers:
            if p.name == args.provider:
                provider = dataclasses.replace(p)

    if "base_url" in args and args.base_url:
        provider = dataclasses.replace(provider, base_url=args.base_url)

        if not provider.name:
            provider.name = args.base_url

        if not provider.display_name:
            provider.display_name = args.base_url

    # find API key
    if "api_key" in args and args.api_key:
        provider.api_key = args.api_key

    if not provider.api_key and provider.api_key_env_var:
        provider.api_key = os.getenv(provider.api_key_env_var)

    return provider
