"""
Batch processing functionality for OpenAI API requests
"""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path

from openai.types import EmbeddingCreateParams
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming

from .providers import get_provider_by_model


class BatchType(Enum):
    CHAT_COMPLETION = "chat_completion"
    EMBEDDING = "embedding"


class Batch:
    def __init__(self, submission_input_file=None, custom_id_prefix="line"):
        if submission_input_file is None:
            # Generate default filename with date and time
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            submission_input_file = f"batch_submission_{current_time}.jsonl"

        # If submission_input_file is a string, create the file
        if isinstance(submission_input_file, str):
            path = Path(submission_input_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.submission_input_file = open(path, "w", encoding="utf-8")
            self._should_close = True
        else:
            self.submission_input_file = submission_input_file
            self._should_close = False

        self.custom_id_prefix = custom_id_prefix
        self.n_bytes = 0
        self.n_requests = 0
        self.model = None
        self.provider = None
        self.batch_type = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._should_close:
            self.submission_input_file.close()
        else:
            self.submission_input_file.flush()

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
