from importlib.metadata import version, PackageNotFoundError

from .batch import Batch
from .providers import get_provider_by_base_url
from ._utils import data_url


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
    b = Batch()
    b.provider = get_provider_by_base_url(client.base_url)
    b.provider.api_key = client.api_key
    b.batch_id = batch_id

    # Implement wait logic using status
    from .batch import FINISHED_STATES

    while True:
        # The dry_run parameter will be passed through kwargs if present
        completed_batch = b.status(**kwargs)

        if callback is not None:
            callback(completed_batch)

        print(completed_batch.status)
        if completed_batch.status in FINISHED_STATES:
            break

        time.sleep(interval)

    # Then download the results to maintain the original behavior
    b.download(**kwargs)

    # Return the completed batch object
    return completed_batch


try:
    __version__ = version("openai_batch")
except PackageNotFoundError:
    # package is not installed
    # Use an editable install (via `pip install -e .`)
    __version__ = "unknown"
