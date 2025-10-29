import base64
import io
import mimetypes
import platform
from pathlib import Path
import urllib.parse
import urllib.request
from typing import Any, Optional, Union, Tuple


def _guess_mime_type(binary: bytes) -> Optional[str]:
    try:
        # noinspection PyUnresolvedReferences
        import filetype

        kind = filetype.guess(binary)
        if kind is not None:
            return kind.mime
    except ImportError:
        pass

    try:
        # Available Python 3.12 or older
        # noinspection PyUnresolvedReferences
        import imghdr

        kind = imghdr.what(None, h=binary)
        if kind:
            return f"image/{kind}"
    except ImportError:
        pass

    return None


def _is_url(string):
    try:
        result = urllib.parse.urlparse(string)
        return all([result.scheme in ("http", "https"), result.netloc])
    except ValueError:
        return False


def _download_from_url(url) -> Tuple[str, bytes]:
    with urllib.request.urlopen(url) as response:
        mime_type = response.getheader("Content-Type")
        data = response.read()

        return mime_type, data


def _save_pil(obj: Any) -> Tuple[Optional[str], Optional[bytes]]:
    try:
        import PIL.Image

        if not isinstance(obj, PIL.Image.Image):
            return None, None

        bio = io.BytesIO()
        obj.save(bio, "JPEG", quality=95)
        return "image/jpeg", bio.getvalue()
    except ImportError:
        return None, None


def _uri_to_path(uri: str) -> Optional[Path]:
    if platform.system() == "Windows" and (uri.startswith("/") or uri.startswith("\\")):
        return Path(uri[1:])  # strip leading slash on Windows

    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme == "file":
        path = urllib.parse.unquote(parsed.path)
        return Path(path)

    return None


def data_url(arg: Union[str, Path, io.RawIOBase, Any]) -> str:
    """
    Helper function to load images into a data url for inlining in prompts.

    Most multimodal models accept input images in two ways.
      * HTTP or HTTPS URL, where the provider will download the image at inference time. This requires hosting the image somewhere where the inference provider can fetch from.
      * Bundled inline as a Base64-encoded data URL. This makes the prompt self-contained and is the preferred method.

    This method converts many possible images into data URLs.

    :param arg: Permissive argument. Can be a path to an image file, URL to download from, a PIL.Image object, bytes.
    :return: a data URL string
    """
    mime = None
    binary = None

    if not arg:
        raise ValueError("Empty argument")

    if isinstance(arg, bytes):
        binary = arg

    if isinstance(arg, str):
        if arg.startswith("data:"):
            return arg

        if _is_url(arg):
            mime, binary = _download_from_url(arg)
        elif Path(arg).exists():
            arg = Path(arg)
        elif uri_path := _uri_to_path(arg):
            arg = uri_path
        else:
            raise ValueError("String does not point to a file or URL: " + arg[:300])

    if isinstance(arg, Path):
        if not arg.is_file():
            raise ValueError("Path must be a file: " + str(arg)[:300])

        mime = mimetypes.guess_type(arg)[0]
        binary = arg.read_bytes()

    if not binary and isinstance(arg, (io.BufferedIOBase, io.RawIOBase, io.BytesIO)):
        binary = arg.read()

    if not binary:
        # noinspection PyBroadException
        try:
            data = arg.read()
            if isinstance(data, bytes):
                binary = data
        except Exception:
            pass

    if not binary:
        # See if it's a PIL image
        mime, binary = _save_pil(arg)

    if not binary:
        raise ValueError("Unknown file type.")

    # Guess the mime type
    if not mime:
        mime = _guess_mime_type(binary)
    if not mime:
        mime = "application/octet-stream"

    # Encode as base64
    encoded = base64.b64encode(binary).decode("utf-8")
    return f"data:{mime};base64,{encoded}"
