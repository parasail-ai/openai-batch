from __future__ import annotations

import time
from typing import Any, Callable

import httpx
import openai
from openai.types.batch import Batch
from openai import NOT_GIVEN

try:
    # noinspection PyProtectedMember
    from openai._types import NotGiven, Body, Query, Headers
except ImportError:
    NotGiven = Any
    Body = Any
    Query = Any
    Headers = Any

FINISHED_STATES = ("failed", "completed", "expired", "cancelled")


def wait(
    client: openai.Client,
    batch_id: str,
    interval: float = 60,
    callback: Callable[[Batch], Any] = None,
    # Extras passed directly to the OpenAI client
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
) -> Batch:
    """
    Wait for batch to complete.
    """

    while True:
        batch = client.batches.retrieve(
            batch_id=batch_id,
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
