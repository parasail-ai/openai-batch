import argparse
import os
from argparse import Namespace
import dataclasses
import typing
from dataclasses import dataclass
import openai.types
from openai import OpenAI


@dataclass
class Provider:
    name: str = None
    display_name: str = None
    base_url: str = None

    api_key: str = None
    api_key_env_var: str = None

    batch_input_max_requests: int = 50_000
    batch_input_max_bytes: int = 100 * 1024 * 1024

    default_chat_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    default_embedding_model: str = "intfloat/e5-mistral-7b-instruct"

    requires_consistency: bool = True  # Default to True for safety

    # Cache for the OpenAI client
    _client = None

    def __str__(self):
        return self.display_name or self.name or self.base_url

    def get_client(self):
        """
        Returns an OpenAI client configured with this provider's base_url and api_key.
        Reuses the client instance if one has already been created.
        """
        if self._client is None:
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client


openai_provider = Provider(
    name="openai",
    display_name="OpenAI",
    base_url="https://api.openai.com/v1",
    api_key_env_var="OPENAI_API_KEY",
    default_chat_model="gpt-4o-mini",
    default_embedding_model="text-embedding-3-small",
    requires_consistency=True,  # OpenAI requires model consistency
)


parasail_provider = Provider(
    name="parasail",
    display_name="Parasail",
    base_url="https://api.parasail.io/v1",
    api_key_env_var="PARASAIL_API_KEY",
    default_chat_model="meta-llama/Meta-Llama-3-8B-Instruct",
    default_embedding_model="intfloat/e5-mistral-7b-instruct",
    requires_consistency=False,  # Parasail allows mixing models
    batch_input_max_bytes=500 * 1024 * 1024,
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


openai_models = list(typing.get_args(openai.types.EmbeddingModel))
openai_models += list(typing.get_args(openai.types.ChatModel))


def get_provider_by_name(name: str) -> Provider:
    """
    Returns the provider with the given name.
    Raises ValueError if no provider with the given name is found.
    """
    for provider in all_providers:
        if provider.name == name:
            provider_copy = dataclasses.replace(provider)
            provider_copy.api_key = os.environ.get(provider_copy.api_key_env_var)
            return provider_copy
    raise ValueError(f"No provider found with name: {name}")


def get_provider_by_model(model: str) -> Provider:

    # If model is in OpenAI's list, use OpenAI provider
    if model in openai_models:
        provider = dataclasses.replace(openai_provider)
        provider.api_key = os.environ.get("OPENAI_API_KEY")
    else:
        provider = dataclasses.replace(parasail_provider)
        provider.api_key = os.environ.get("PARASAIL_API_KEY")
    return provider


def get_provider_by_base_url(base_url: str) -> Provider:
    """
    Returns the appropriate provider based on the given base URL.
    If the base URL matches a known provider, returns that provider.
    Otherwise, returns a new provider with the given base URL.
    """
    # Check if the base URL matches any of the known providers
    for provider in all_providers:
        if provider.base_url == base_url:
            return dataclasses.replace(provider)

    # If no match found, create a new provider with the given base URL
    return Provider(base_url=base_url)


def _get_provider(args: Namespace) -> Provider:
    provider = Provider()

    if args:
        if args.provider:
            for p in all_providers:
                if p.name == args.provider:
                    provider = dataclasses.replace(p)

        if "base_url" in args and args.base_url:
            provider = dataclasses.replace(provider, base_url=args.base_url)

        # find API key
        if "api_key" in args and args.api_key:
            provider.api_key = args.api_key

        if not provider.api_key and provider.api_key_env_var:
            provider.api_key = os.getenv(provider.api_key_env_var)

    return provider
