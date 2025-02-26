from importlib.metadata import version, PackageNotFoundError

from .batch import Batch
from .providers import _get_provider


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
    batch.provider = _get_provider({"base_url": client.base_url, "api_key": client.api_key})
    batch.batch_id = batch_id

    return batch.wait(*args[2:], **kwargs)


try:
    __version__ = version("openai_batch")
except PackageNotFoundError:
    # package is not installed
    # Use an editable install (via `pip install -e .`)
    __version__ = "unknown"
