from importlib.metadata import version, PackageNotFoundError

from ._utils import wait

try:
    __version__ = version("openai_batch")
except PackageNotFoundError:
    # package is not installed
    # Use an editable install (via `pip install -e .`)
    __version__ = "unknown"
