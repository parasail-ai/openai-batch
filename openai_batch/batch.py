"""
Batch processing functionality for OpenAI API requests
"""

import json
import time
from typing import Any, Callable, Optional
import httpx
from openai import OpenAI, NOT_GIVEN
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

from .providers import get_provider_by_model

FINISHED_STATES = ("failed", "completed", "expired", "cancelled")


class BatchType(Enum):
    CHAT_COMPLETION = "chat_completion"
    EMBEDDING = "embedding"


class Batch:
    def __init__(
        self, submission_input_file=None, output_file=None, error_file=None, custom_id_prefix="line"
    ):
        self.submission_input_file = submission_input_file
        self.output_file = output_file
        self.error_file = error_file
        self.custom_id_prefix = custom_id_prefix

        self.submission_file = None
        self._should_close = False
        self.n_bytes = 0
        self.n_requests = 0
        self.model = None
        self.provider = None
        self.batch_type = None
        self.batch_id = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.submission_file and self._should_close:
            self.submission_file.close()

    def _ensure_submission_file(self):
        """Ensure submission file is ready for writing"""
        if self.submission_file is not None:
            return

        # Generate default filename if none provided
        if self.submission_input_file is None:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.submission_input_file = f"batch_submission_{current_time}.jsonl"

        # Create file if path provided
        if isinstance(self.submission_input_file, (str, Path)):
            path = Path(self.submission_input_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.submission_file = open(path, "w", encoding="utf-8")
            self._should_close = True
        # Use bytes directly
        elif isinstance(self.submission_input_file, bytes):
            from io import BytesIO

            self.submission_file = BytesIO(self.submission_input_file)
            self._should_close = True
        else:
            # Assume it's a file-like object
            self.submission_file = self.submission_input_file
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
        self.submission_file.write(line)
        self.n_bytes += n_bytes
        self.n_requests += 1

    def add_to_batch(self, **kwargs):
        # Ensure model is included in kwargs
        if "model" not in kwargs:
            raise ValueError("Model must be specified in arguments")

        # Determine request type based on kwargs
        is_embedding = "input" in kwargs
        is_chat_completion = "messages" in kwargs

        if not is_embedding and not is_chat_completion:
            raise ValueError(
                "Request must include either 'input' for embeddings or 'messages' for chat completions"
            )
        if is_embedding and is_chat_completion:
            raise ValueError("Request cannot include both 'input' and 'messages'")

        # On first request, determine provider and batch type
        if self.provider is None:
            self.model = kwargs["model"]
            self.provider = get_provider_by_model(self.model)
            self.batch_type = BatchType.EMBEDDING if is_embedding else BatchType.CHAT_COMPLETION
        else:
            # Validate batch type matches request type
            if is_embedding and self.batch_type != BatchType.EMBEDDING:
                raise ValueError("Cannot add embedding to a chat completion batch")
            if is_chat_completion and self.batch_type != BatchType.CHAT_COMPLETION:
                raise ValueError("Cannot add chat completion to an embedding batch")
            if self.provider.requires_consistency and self.model != kwargs["model"]:
                raise ValueError(
                    f"Model mismatch. Provider {self.provider.name} requires model consistency. "
                    f"First request determines model for batch and is {self.model} and subsequent request is {kwargs['model']}"
                )

        # Create appropriate body and add to batch
        if is_embedding:
            body = EmbeddingCreateParams(**kwargs)
            self._add_to_batch(body, "/v1/embeddings")
        else:  # is_chat_completion
            body = CompletionCreateParamsNonStreaming(**kwargs)
            self._add_to_batch(body, "/v1/chat/completions")

    def submit(self) -> str:
        """
        Submit the batch job using the current submission file.

        :return: The batch ID
        """
        if not self.provider:
            raise ValueError("No requests have been added to the batch")

        # Create OpenAI client
        client = OpenAI(base_url=self.provider.base_url, api_key=self.provider.api_key)

        # Close and prepare submission file for reading
        if isinstance(self.submission_file, (BytesIO, str, Path)):
            if self._should_close:
                self.submission_file.close()

            if isinstance(self.submission_file, BytesIO):
                self.submission_file.seek(0)
                file_content = self.submission_file.read()
            else:
                with open(self.submission_input_file, "rb") as f:
                    file_content = f.read()

            input_file = client.files.create(file=file_content, purpose="batch")
        else:
            # File-like object provided by user
            input_file = client.files.create(file=self.submission_file, purpose="batch")

        # Create batch
        batch = client.batches.create(
            input_file_id=input_file.id,
            completion_window="24h",
            endpoint="/v1/chat/completions",
        )

        self.batch_id = batch.id
        return self.batch_id

    def wait(
        self,
        interval: float = 60,
        callback: Callable[[OpenAIBatch], Any] = None,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> OpenAIBatch:
        """
        Wait for the batch to complete.

        :param interval: How long to wait between each poll (in seconds)
        :param callback: Called after each API retrieve
        :param extra_headers: Forwarded to OpenAI client
        :param extra_query: Forwarded to OpenAI client
        :param extra_body: Forwarded to OpenAI client
        :param timeout: Forwarded to OpenAI client
        :return: The completed batch object
        """
        if not self.batch_id:
            raise ValueError("Batch has not been submitted yet")

        client = OpenAI(base_url=self.provider.base_url, api_key=self.provider.api_key)

        while True:
            batch = client.batches.retrieve(
                batch_id=self.batch_id,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            )

            if callback is not None:
                callback(batch)

            if batch.status in FINISHED_STATES:
                # Download output file if present
                if batch.output_file_id:
                    contents = client.files.content(batch.output_file_id).content
                    output_path = self.output_file or f"{self.batch_id}-output.jsonl"
                    Path(output_path).write_bytes(contents)

                # Download error file if present
                if batch.error_file_id:
                    contents = client.files.content(batch.error_file_id).content
                    error_path = self.error_file or f"{self.batch_id}-errors.jsonl"
                    Path(error_path).write_bytes(contents)

                return batch

            time.sleep(interval)

    def submit_and_wait(
        self,
        interval: float = 60,
        callback: Callable[[OpenAIBatch], Any] = None,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> OpenAIBatch:
        """
        Submit the batch and wait for it to complete.

        :param interval: How long to wait between each poll (in seconds)
        :param callback: Called after each API retrieve
        :param extra_headers: Forwarded to OpenAI client
        :param extra_query: Forwarded to OpenAI client
        :param extra_body: Forwarded to OpenAI client
        :param timeout: Forwarded to OpenAI client
        :return: The completed batch object
        """
        self.submit()
        return self.wait(
            interval=interval,
            callback=callback,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )
