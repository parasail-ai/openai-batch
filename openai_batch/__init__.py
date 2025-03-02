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
    Deprecated: Use Batch.wait() instead.
    This function is maintained for backward compatibility.
    """
    import warnings

    warnings.warn(
        "The wait() function is deprecated. Use Batch.wait() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Create a batch object and call its wait method
    client = args[0] if args else kwargs.pop("client")
    batch_id = args[1] if len(args) > 1 else kwargs.pop("batch_id")

    # Create batch object for resuming
    batch = Batch()
    batch.provider = get_provider_by_base_url(client.base_url)
    batch.provider.api_key = client.api_key
    batch.batch_id = batch_id

    # The dry_run parameter will be passed through kwargs if present
    return batch.wait(*args[2:], **kwargs)


try:
    __version__ = version("openai_batch")
except PackageNotFoundError:
    # package is not installed
    # Use an editable install (via `pip install -e .`)
    __version__ = "unknown"
