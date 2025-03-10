"""
Batch processing functionality for OpenAI API requests
"""

import json
import time
from typing import Any, Callable, Optional, Union, Tuple
import httpx
from io import BytesIO, TextIOWrapper
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

    def submit(self, dry_run: bool = False) -> str:
        """
        Submit the batch job using the current submission file.

        :param dry_run: If True, skip actual API calls and return a mock batch ID (for testing)
        :return: The batch ID
        """
        if not self.provider:
            raise ValueError("No requests have been added to the batch")

        # If dry_run is enabled, return a mock batch ID without making API calls
        if dry_run:
            self.batch_id = "batch-dry-run"
            return self.batch_id

        # Create OpenAI client
        client = OpenAI(base_url=self.provider.base_url, api_key=self.provider.api_key)

        # Close and prepare submission file for reading
        if isinstance(self.submission_input_file, (TextIOWrapper, BytesIO, str, Path)):
            if self._should_close:
                self.submission_input_file.close()

            if isinstance(self.submission_input_file, BytesIO):
                self.submission_input_file.seek(0)
                file_content = self.submission_input_file.read()
            else:
                if isinstance(self.submission_input_file, (str, Path)):
                    file_path = Path(self.submission_input_file)
                elif isinstance(self.submission_input_file, TextIOWrapper):
                    file_path = Path(self.submission_input_file.name)
                with open(file_path, "rb") as f:
                    file_content = f.read()

            input_file = client.files.create(file=file_content, purpose="batch")
        else:
            # File-like object provided by user
            input_file = client.files.create(file=self.submission_input_file, purpose="batch")

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
        extra_headers: Optional[Headers] = None,
        extra_query: Optional[Query] = None,
        extra_body: Optional[Body] = None,
        timeout: Union[float, httpx.Timeout, None, NotGiven] = NOT_GIVEN,
        dry_run: bool = False,
    ) -> OpenAIBatch:
        """
        Wait for the batch to complete.

        :param interval: How long to wait between each poll (in seconds)
        :param callback: Called after each API retrieve
        :param extra_headers: Forwarded to OpenAI client
        :param extra_query: Forwarded to OpenAI client
        :param extra_body: Forwarded to OpenAI client
        :param timeout: Forwarded to OpenAI client
        :param dry_run: If True, skip actual API calls and return a mock batch object (for testing)
        :return: The completed batch object
        """
        if not self.batch_id:
            raise ValueError("Batch has not been submitted yet")

        # If dry_run is enabled, return a mock batch object without making API calls
        if dry_run:
            # Create a mock batch object
            mock_batch = OpenAIBatch(
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

            # Call the callback if provided
            if callback is not None:
                callback(mock_batch)

            return mock_batch

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
                return batch

            time.sleep(interval)

    def download(
        self,
        batch: Optional[OpenAIBatch] = None,
        extra_headers: Optional[Headers] = None,
        extra_query: Optional[Query] = None,
        extra_body: Optional[Body] = None,
        timeout: Union[float, httpx.Timeout, None, NotGiven] = NOT_GIVEN,
        dry_run: bool = False,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Download output and error files for a completed batch.

        :param batch: The batch object to download files for (uses self.batch_id if None)
        :param extra_headers: Forwarded to OpenAI client
        :param extra_query: Forwarded to OpenAI client
        :param extra_body: Forwarded to OpenAI client
        :param timeout: Forwarded to OpenAI client
        :param dry_run: If True, skip actual API calls and create empty files (for testing)
        :return: Tuple of (output_path, error_path) with the paths to the downloaded files
        """
        if not self.batch_id and batch is None:
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

        client = OpenAI(base_url=self.provider.base_url, api_key=self.provider.api_key)

        # Use provided batch object or retrieve the current batch
        if batch is None:
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
        callback: Callable[[OpenAIBatch], Any] = None,
        extra_headers: Optional[Headers] = None,
        extra_query: Optional[Query] = None,
        extra_body: Optional[Body] = None,
        timeout: Union[float, httpx.Timeout, None, NotGiven] = NOT_GIVEN,
        dry_run: bool = False,
    ) -> Tuple[OpenAIBatch, Optional[str], Optional[str]]:
        """
        Submit the batch, wait for it to complete, and download the results.

        :param interval: How long to wait between each poll (in seconds)
        :param callback: Called after each API retrieve
        :param extra_headers: Forwarded to OpenAI client
        :param extra_query: Forwarded to OpenAI client
        :param extra_body: Forwarded to OpenAI client
        :param timeout: Forwarded to OpenAI client
        :param dry_run: If True, skip actual API calls and return mock objects (for testing)
        :return: Tuple of (batch, output_path, error_path)
        """
        self.submit(dry_run=dry_run)
        batch = self.wait(
            interval=interval,
            callback=callback,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            dry_run=dry_run,
        )
        output_path, error_path = self.download(
            batch=batch,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            dry_run=dry_run,
        )
        return batch, output_path, error_path
