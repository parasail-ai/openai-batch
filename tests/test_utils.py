import platform
from io import BytesIO
from urllib.error import HTTPError
from pathlib import Path
import PIL.Image

import pytest

from openai_batch import data_url


def test_image_to_data_url():
    jpg_path = Path(__file__).parent / "resources" / "example.jpg"
    jpg_bytes = jpg_path.read_bytes()

    # Test Path object
    data = data_url(jpg_path)
    assert isinstance(data, str)
    assert len(data) > 100
    assert data.startswith("data:image/jpeg;base64,")
    expected = data

    # Test plain string path
    data = data_url(str(jpg_path))
    assert data == expected

    # Test local URI
    if platform.system() != "Windows":
        data = data_url(jpg_path.as_uri())
        assert data == expected

    # Test open file object
    with open(jpg_path, "rb") as f:
        data = data_url(f)
        assert data == expected

    # Test bytes
    data = data_url(jpg_bytes)
    assert data == expected

    # Test BytesIO
    data = data_url(BytesIO(jpg_bytes))
    assert data == expected

    # Test PIL
    data = data_url(PIL.Image.open(jpg_path))
    assert isinstance(data, str)
    assert len(data) > 100
    assert data.startswith("data:image/jpeg;base64,")

    # Test invalid
    pytest.raises(ValueError, data_url, None)
    pytest.raises(ValueError, data_url, "abc")
    pytest.raises(ValueError, data_url, str(jpg_path) + "does-not-exist")
    pytest.raises(ValueError, data_url, Path(__file__).parent)


@pytest.mark.skip(reason="Requires a network connection")
def test_image_to_data_url_network():
    # Valid URL
    url = "https://picsum.photos/100/100"
    data = data_url(url)
    assert isinstance(data, str)
    assert len(data) > 100
    assert data.startswith("data:image/jpeg;base64,")

    # Invalid URL
    pytest.raises(HTTPError, data_url, "https://example.com/does-not-exist.jpg")
