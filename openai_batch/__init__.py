from importlib.metadata import version, PackageNotFoundError

from .batch import Batch
from .providers import get_provider_by_base_url


# Backward compatibility
# Original wait definition:
# def wait(
#     client: openai.Client,
#     batch_id: Iterable[str] | str,
#     interval: float = 60,
#     callback: Callable[[Batch], Any] = None,
#     finished_callback: Callable[[Batch], Any] = None,
#     # Extras passed directly to the OpenAI client
#     extra_headers: Headers | None = None,
#     extra_query: Query | None = None,
#     extra_body: Body | None = None,
#     timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
# )
# batch.wait definition:
# def wait(
#     self,
#     interval: float = 60,
#     callback: Callable[[OpenAIBatch], Any] = None,
#     extra_headers: Headers | None = None,
#     extra_query: Query | None = None,
#     extra_body: Body | None = None,
#     timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
# ) -> OpenAIBatch:
# Therefore need to figure out the provider from the OpenAI client object, then create a batch


def wait(*args, **kwargs):
    """
    Deprecated: Use Batch.status(), Batch.submit_wait_download() and Batch.download() instead.
    This function is maintained for backward compatibility.
    """
    import warnings
    import time

    warnings.warn(
        "The wait() function is deprecated. Use Batch.status(), Batch.submit_wait_download() and Batch.download() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Create a batch object and implement wait logic
    client = args[0] if args else kwargs.pop("client")
    batch_id = args[1] if len(args) > 1 else kwargs.pop("batch_id")

    # Extract interval parameter if present
    interval = kwargs.pop("interval", 60) if len(args) <= 2 else args[2]

    # Extract callback parameter if present
    callback = kwargs.pop("callback", None) if len(args) <= 3 else args[3]

    # Create batch object for resuming
    batch = Batch()
    batch.provider = get_provider_by_base_url(client.base_url)
    batch.provider.api_key = client.api_key
    batch.batch_id = batch_id

    # Implement wait logic using status
    from .batch import FINISHED_STATES

    completed_batch = None
    while True:
        # The dry_run parameter will be passed through kwargs if present
        completed_batch = batch.status(**kwargs)

        if callback is not None:
            callback(completed_batch)

        print(completed_batch.status)
        if completed_batch.status in FINISHED_STATES:
            break

        time.sleep(interval)

    # Then download the results to maintain the original behavior
    batch.download(batch=completed_batch, **kwargs)

    # Return the completed batch object
    return completed_batch


try:
    __version__ = version("openai_batch")
except PackageNotFoundError:
    # package is not installed
    # Use an editable install (via `pip install -e .`)
    __version__ = "unknown"
