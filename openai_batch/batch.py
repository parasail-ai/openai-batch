"""
Batch processing functionality for OpenAI API requests
"""

import json
import os
import time
import dataclasses
from typing import Any, Callable, Optional, Union, Tuple
import httpx
from io import BytesIO, TextIOWrapper
from openai import NOT_GIVEN
from openai.types.batch import Batch as OpenAIBatch
from datetime import datetime
from enum import Enum
from pathlib import Path

try:
    # noinspection PyProtectedMember
    from openai._types import NotGiven, Body, Query, Headers
except ImportError:
    NotGiven = Any
    Body = Any
    Query = Any
    Headers = Any

from openai.types import EmbeddingCreateParams
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming

from .providers import get_provider_by_model, all_providers

FINISHED_STATES = ("failed", "completed", "expired", "cancelled")


class BatchType(Enum):
    CHAT_COMPLETION = "chat_completion"
    EMBEDDING = "embedding"
    SCORE = "score"
    RERANK = "rerank"
    TRANSFUSION = "transfusion"


class Batch:
    def __init__(
        self,
        submission_input_file=None,
        output_file=None,
        error_file=None,
        custom_id_prefix="line",
        batch_id=None,
        provider=None,
    ):
        if submission_input_file is not None and batch_id is not None:
            raise ValueError(
                "Cannot specify both submission_input_file and batch_id because adding to an existing batch is not supported"
            )

        self.submission_input_file = submission_input_file
        self.output_file = output_file
        self.error_file = error_file
        self.custom_id_prefix = custom_id_prefix
        self.batch_id = batch_id

        self._should_close = False
        self.n_bytes = 0
        self.n_requests = 0
        self.model = None
        self.provider = provider
        self.batch_type = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.submission_input_file and self._should_close:
            self.submission_input_file.close()

    def _ensure_submission_file(self):
        """Ensure submission file is ready for writing"""

        # Generate default filename if none provided
        if self.submission_input_file is None:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.submission_input_file = f"batch_submission_{current_time}.jsonl"

        # Create file if path provided
        if isinstance(self.submission_input_file, (str, Path)):
            path = Path(self.submission_input_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.submission_input_file = open(path, "w", encoding="utf-8")
            self._should_close = True
        # Use bytes directly
        elif isinstance(self.submission_input_file, bytes):
            self.submission_input_file = BytesIO(self.submission_input_file)
            self._should_close = True

        elif isinstance(self.submission_input_file, TextIOWrapper):
            self._should_close = True
        else:
            self._should_close = False

    def _get_custom_id(self):
        return f"{self.custom_id_prefix}-{self.n_requests + 1}"

    def _add_to_batch(self, body, url):
        if self.n_requests >= self.provider.batch_input_max_requests:
            raise ValueError(
                f"Exceeded max number of requests per batch ({self.provider.batch_input_max_requests})"
            )

        request = {
            "custom_id": self._get_custom_id(),
            "method": "POST",
            "url": url,
            "body": body,
        }

        line = json.dumps(request) + "\n"
        n_bytes = len(line.encode("utf-8"))

        if self.n_bytes + n_bytes > self.provider.batch_input_max_bytes:
            raise ValueError(
                f"Exceeded max batch input file size ({self.provider.batch_input_max_bytes // 1024 // 1024} MB)"
            )

        self._ensure_submission_file()
        self.submission_input_file.write(line)
        self.n_bytes += n_bytes
        self.n_requests += 1

    def add_to_batch(self, **kwargs):
        # Check if batch_id is set, which means we're working with an existing batch
        if self.batch_id is not None:
            raise ValueError("Adding to an existing batch is not supported")

        # Ensure model is included in kwargs
        if "model" not in kwargs:
            raise ValueError("Model must be specified in arguments")

        # Determine request type based on kwargs
        is_embedding = "input" in kwargs
        is_chat_completion = "messages" in kwargs
        is_rerank = "documents" in kwargs
        is_score = "text_1" in kwargs
        is_transfusion = kwargs["model"] == "Shitao/OmniGen-v1" or (
            "prompt" in kwargs and "size" in kwargs and "image" in kwargs
        )

        # Validate request type
        request_type_count = sum(
            [is_embedding, is_chat_completion, is_rerank, is_score, is_transfusion]
        )
        if request_type_count == 0:
            raise ValueError(
                "Request must include either 'input' for embeddings, 'messages' for chat completions, "
                "'documents' for rerank, or 'text_1' for score"
            )
        if request_type_count > 1:
            raise ValueError(
                "Request cannot include multiple types of parameters. Use only one of: "
                "'input', 'messages', 'documents', or 'text_1'"
            )

        # Set batch type if not already set
        if self.batch_type is None:
            if is_embedding:
                self.batch_type = BatchType.EMBEDDING
            elif is_chat_completion:
                self.batch_type = BatchType.CHAT_COMPLETION
            elif is_score:
                self.batch_type = BatchType.SCORE
            elif is_transfusion:
                self.batch_type = BatchType.TRANSFUSION
            else:  # is_rerank
                self.batch_type = BatchType.RERANK

        # Set model if not already set
        if self.model is None:
            self.model = kwargs["model"]

        # On first request, determine provider
        if self.provider is None:
            self.provider = get_provider_by_model(self.model)
        else:
            # Validate batch type matches request type
            if is_embedding and self.batch_type != BatchType.EMBEDDING:
                raise ValueError(f"Cannot add embedding to a {self.batch_type.value} batch")
            if is_chat_completion and self.batch_type != BatchType.CHAT_COMPLETION:
                raise ValueError(f"Cannot add chat completion to a {self.batch_type.value} batch")
            if is_score and self.batch_type != BatchType.SCORE:
                raise ValueError(f"Cannot add score request to a {self.batch_type.value} batch")
            if is_rerank and self.batch_type != BatchType.RERANK:
                raise ValueError(f"Cannot add rerank request to a {self.batch_type.value} batch")
            if is_transfusion and self.batch_type != BatchType.TRANSFUSION:
                raise ValueError(
                    f"Cannot add transfusion request to a {self.batch_type.value} batch"
                )
            if self.provider.requires_consistency and self.model != kwargs["model"]:
                raise ValueError(
                    f"Model mismatch. Provider {self.provider.name} requires model consistency. "
                    f"First request determines model for batch and is {self.model} and subsequent request is {kwargs['model']}"
                )

        # Create appropriate body and add to batch
        if is_embedding:
            body = EmbeddingCreateParams(**kwargs)
            self._add_to_batch(body, "/v1/embeddings")
        elif is_chat_completion:
            body = CompletionCreateParamsNonStreaming(**kwargs)
            self._add_to_batch(body, "/v1/chat/completions")
        elif is_score:
            # Use the raw kwargs as the body since there's no specific parameter class for score
            if "text_2" not in kwargs:
                raise ValueError("'text_2' is required for score requests")

            body = {
                "model": kwargs["model"],
                "text_1": kwargs["text_1"],
                "text_2": kwargs["text_2"],
            }
            self._add_to_batch(body, "/v1/score")
        elif is_transfusion:
            # Verify all required parameters are present
            required_params = ["prompt", "size", "image", "response_format"]
            missing_params = [param for param in required_params if param not in kwargs]
            if missing_params:
                raise ValueError(
                    f"Missing required parameters for transfusion requests: {', '.join(missing_params)}"
                )

            # Use the raw kwargs as the body since there's no specific parameter class for transfusion
            body = kwargs.copy()

            # if kwargs["image"] is a string, make it a list
            if isinstance(kwargs.get("image"), str):
                body["image"] = [kwargs["image"]]

            self._add_to_batch(body, "/v1/images/edits")
        else:  # is_rerank
            # Use the raw kwargs as the body since there's no specific parameter class for rerank
            if isinstance(kwargs.get("documents"), str):
                kwargs["documents"] = [kwargs["documents"]]

            if not isinstance(kwargs.get("documents"), list):
                raise ValueError("Rerank 'documents' must be a list of strings.")

            if "query" not in kwargs:
                raise ValueError("'query' is required for rerank requests")

            rerank_request_param_whitelist = [
                "model",
                "query",
                "documents",
                "top_n",
                "priority",
                "truncate_prompt_tokens",
            ]

            body = {k: kwargs[k] for k in kwargs.keys() if k in rerank_request_param_whitelist}
            self._add_to_batch(body, "/v1/score")

    def submit(self, metadata: Optional[dict] = None, dry_run: bool = False) -> str:
        """
        Submit the batch job using the current submission file.

        :param metadata: Optional metadata to associate with the batch
        :param dry_run: If True, skip actual API calls and return a mock batch ID (for testing)
        :return: The batch ID
        """
        if not self.provider:
            raise ValueError("No requests have been added to the batch")

        # If dry_run is enabled, return a mock batch ID without making API calls
        if dry_run:
            self.batch_id = "batch-dry-run"
            return self.batch_id

        # Get OpenAI client from provider
        client = self.provider.get_client()

        # Close and prepare submission file for reading
        if isinstance(self.submission_input_file, (TextIOWrapper, BytesIO, str, Path)):
            if self._should_close:
                self.submission_input_file.close()

            if isinstance(self.submission_input_file, BytesIO):
                self.submission_input_file.seek(0)
                file_content = self.submission_input_file.read()
            elif isinstance(self.submission_input_file, TextIOWrapper):
                file_content = self.submission_input_file.read().encode("utf-8")
            else:
                file_content = Path(self.submission_input_file).read_bytes()

            input_file = client.files.create(file=file_content, purpose="batch")
        else:
            # File-like object provided by user
            input_file = client.files.create(file=self.submission_input_file, purpose="batch")

        # Determine the endpoint based on batch type
        if self.batch_type == BatchType.CHAT_COMPLETION:
            endpoint = "/v1/chat/completions"
        elif self.batch_type == BatchType.EMBEDDING:
            endpoint = "/v1/embeddings"
        elif self.batch_type == BatchType.RERANK:
            endpoint = "/v1/score"
        elif self.batch_type == BatchType.SCORE:
            endpoint = "/v1/score"
        elif self.batch_type == BatchType.TRANSFUSION:
            endpoint = "/v1/images/edits"
        else:
            # Default to chat completions for backward compatibility
            endpoint = "/v1/chat/completions"

        # Create batch
        batch = client.batches.create(
            input_file_id=input_file.id,
            completion_window="24h",
            endpoint=endpoint,
            metadata=metadata,
        )

        self.batch_id = batch.id
        return self.batch_id

    def auto_detect_provider(self):
        """
        Attempt to auto-detect the provider by trying to retrieve the batch status from all available providers.
        If successful, sets the provider object. Otherwise, raises an exception.

        This is used when a batch job is resumed (batch_id is provided but not submission_input_file)
        and the provider is not set.
        """
        # These should be programming errors, not user errors
        assert self.batch_id, "Cannot auto-detect provider without a batch ID"
        assert self.provider is None, "Provider is already set, no need to auto-detect"

        for provider in all_providers:
            try:
                # Create a copy of the provider with the API key
                provider_copy = dataclasses.replace(provider)
                provider_copy.api_key = os.environ.get(provider_copy.api_key_env_var)

                if not provider_copy.api_key:
                    continue

                # Try to retrieve the batch status using this provider
                client = provider_copy.get_client()
                client.batches.retrieve(batch_id=self.batch_id)

                # If we get here, the provider worked
                self.provider = provider_copy
                return
            except Exception:
                # This provider didn't work, try the next one
                continue

        # If we get here, no provider worked
        raise ValueError(
            f"Could not auto-detect provider for batch ID {self.batch_id}. Please specify a provider."
        )

    def status(
        self,
        extra_headers: Optional[Headers] = None,
        extra_query: Optional[Query] = None,
        extra_body: Optional[Body] = None,
        timeout: Union[float, httpx.Timeout, None, NotGiven] = NOT_GIVEN,
        dry_run: bool = False,
    ) -> OpenAIBatch:
        """
        Get the status of the current batch job.

        :param extra_headers: Forwarded to OpenAI client
        :param extra_query: Forwarded to OpenAI client
        :param extra_body: Forwarded to OpenAI client
        :param timeout: Forwarded to OpenAI client
        :param dry_run: If True, skip actual API calls and return a mock batch object (for testing)
        :return: The batch object
        """
        if not self.batch_id:
            raise ValueError("Batch has not been submitted yet")

        # If dry_run is enabled, return a mock batch object without making API calls
        if dry_run:
            # Create a mock batch object
            return OpenAIBatch(
                id=self.batch_id,
                status="completed",
                completion_window="24h",
                created_at=0,
                endpoint="/v1/chat/completions",
                input_file_id="file-dry-run-input",
                output_file_id="file-dry-run-output",
                error_file_id="file-dry-run-error",
                object="batch",
            )
        # Auto-detect provider if not set
        if self.provider is None:
            self.auto_detect_provider()

        client = self.provider.get_client()
        batch = client.batches.retrieve(
            batch_id=self.batch_id,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )
        return batch

    def download(
        self,
        extra_headers: Optional[Headers] = None,
        extra_query: Optional[Query] = None,
        extra_body: Optional[Body] = None,
        timeout: Union[float, httpx.Timeout, None, NotGiven] = NOT_GIVEN,
        dry_run: bool = False,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Download output and error files for a completed batch.

        :param extra_headers: Forwarded to OpenAI client
        :param extra_query: Forwarded to OpenAI client
        :param extra_body: Forwarded to OpenAI client
        :param timeout: Forwarded to OpenAI client
        :param dry_run: If True, skip actual API calls and create empty files (for testing)
        :return: Tuple of (output_path, error_path) with the paths to the downloaded files
        """
        if not self.batch_id:
            raise ValueError("Batch has not been submitted yet")

        # If dry_run is enabled, create empty files without making API calls
        if dry_run:
            output_path = None
            error_path = None

            # Create empty output and error files if paths are provided
            if self.output_file:
                Path(self.output_file).write_text("")
                output_path = self.output_file

            if self.error_file:
                Path(self.error_file).write_text("")
                error_path = self.error_file

            return output_path, error_path

        # Auto-detect provider if not set
        if self.provider is None:
            self.auto_detect_provider()

        client = self.provider.get_client()

        # Use provided batch object or retrieve the current batch
        batch = client.batches.retrieve(
            batch_id=self.batch_id,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )

        output_path = None
        error_path = None

        # Download output file if present
        if batch.output_file_id:
            contents = client.files.content(batch.output_file_id).content
            output_path = self.output_file or f"{batch.id}-output.jsonl"
            Path(output_path).write_bytes(contents)

        # Download error file if present
        if batch.error_file_id:
            contents = client.files.content(batch.error_file_id).content
            error_path = self.error_file or f"{batch.id}-errors.jsonl"
            Path(error_path).write_bytes(contents)

        return output_path, error_path

    def submit_wait_download(
        self,
        interval: float = 60,
        status_callback: Callable[[OpenAIBatch], Any] = None,
        extra_headers: Optional[Headers] = None,
        extra_query: Optional[Query] = None,
        extra_body: Optional[Body] = None,
        timeout: Union[float, httpx.Timeout, None, NotGiven] = NOT_GIVEN,
        metadata: Optional[dict] = None,
        dry_run: bool = False,
    ) -> Tuple[OpenAIBatch, Optional[str], Optional[str]]:
        """
        Submit the batch, wait for it to complete, and download the results.

        :param interval: How long to wait between each poll (in seconds)
        :param status_callback: Called after each API retrieve
        :param extra_headers: Forwarded to OpenAI client
        :param extra_query: Forwarded to OpenAI client
        :param extra_body: Forwarded to OpenAI client
        :param timeout: Forwarded to OpenAI client
        :param metadata: Optional metadata to associate with the batch
        :param dry_run: If True, skip actual API calls and return mock objects (for testing)
        :return: Tuple of (batch, output_path, error_path)
        """
        self.submit(metadata=metadata, dry_run=dry_run)

        # Wait for the batch to complete
        batch = None
        while True:
            batch = self.status(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                dry_run=dry_run,
            )

            if status_callback is not None:
                status_callback(batch)

            print(batch.status)
            if batch.status in FINISHED_STATES:
                break

            time.sleep(interval)

        # Download results
        output_path, error_path = self.download(
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            dry_run=dry_run,
        )
        return batch, output_path, error_path
