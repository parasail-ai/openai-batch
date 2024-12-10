from __future__ import annotations

import time
from typing import Any, Callable, Iterable, List

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
    batch_id: Iterable[str] | str,
    interval: float = 60,
    callback: Callable[[Batch], Any] = None,
    finished_callback: Callable[[Batch], Any] = None,
    # Extras passed directly to the OpenAI client
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
) -> List[Batch] | Batch:
    """
    Wait for one or more batches to complete.

    :param client: OpenAI API client.
    :param batch_id: one or more batch IDs to wait on.
    :param interval: How long to wait between each poll (in seconds).
    :param callback: Called after each API retrieve.
    :param finished_callback: Called once per finished batch.
    :param extra_headers: Forwarded to OpenAI client.
    :param extra_query: Forwarded to OpenAI client.
    :param extra_body: Forwarded to OpenAI client.
    :param timeout: Forwarded to OpenAI client.
    :return: The batch object returned by the OpenAI client, or a list of these if ``batch_id`` is a list.
    """

    single = isinstance(batch_id, str)
    ret_order = [batch_id] if single else list(batch_id)
    batches = {}

    waiting_on = set(ret_order)
    while waiting_on:
        for batch_id in tuple(waiting_on):
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
                if finished_callback is not None:
                    finished_callback(batch)

                waiting_on.remove(batch_id)
                batches[batch_id] = batch

        if waiting_on:
            time.sleep(interval)

    ret = [batches[bid] for bid in ret_order]
    return ret[0] if single else ret
