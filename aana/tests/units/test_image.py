# ruff: noqa: S101, NPY002
import base64
from importlib import resources
from pathlib import Path

import numpy as np
import PIL.Image
import pytest

from aana.core.models.image import Image


def load_numpy_from_image_bytes(content: bytes) -> np.ndarray:
    """Load a numpy array from image bytes."""
    image = Image(content=content, save_on_disk=False)
    return image.get_numpy()


@pytest.fixture
def test_numpy_image():
    """Fixture to create a test numpy image."""
    # Create a random numpy array
    numpy_image = np.random.rand(100, 100, 3).astype(np.uint8)
    return numpy_image


@pytest.fixture
def mock_download_file(mocker):
    """Mock download_file function in both media and image modules."""
    # Path to the file to be used as mock return value
    path = resources.files("aana.tests.files.images") / "Starry_Night.jpeg"
    content = path.read_bytes()

    # Mock for aana.core.models.media.download_file
    mock_media_download = mocker.patch(
        "aana.core.models.media.download_file", autospec=True
    )
    mock_media_download.return_value = content

    # Mock for aana.core.models.image.download_file
    mock_image_download = mocker.patch(
        "aana.core.models.image.download_file", autospec=True
    )
    mock_image_download.return_value = content

    # Return both mocks in case you need them in your tests
    return mock_media_download, mock_image_download


def test_image(mock_download_file):
    """Test that the image can be created from path, url, content, or numpy."""
    try:
        path = resources.files("aana.tests.files.images") / "Starry_Night.jpeg"
        image = Image(path=path, save_on_disk=False)
        assert image.path == path
        assert image.content is None
        assert image.numpy is None
        assert image.url is None

        numpy = image.get_numpy()
        assert numpy.shape == (720, 909, 3)

        content = image.get_content()
        numpy = load_numpy_from_image_bytes(content)
        assert numpy.shape == (720, 909, 3)
    finally:
        image.cleanup()

    try:
        url = "http://example.com/Starry_Night.jpeg"
        image = Image(url=url, save_on_disk=False)
        assert image.path is None
        assert image.content is None
        assert image.numpy is None
        assert image.url == url

        numpy = image.get_numpy()
        assert numpy.shape == (720, 909, 3)

        content = image.get_content()
        numpy = load_numpy_from_image_bytes(content)
        assert numpy.shape == (720, 909, 3)
    finally:
        image.cleanup()

    try:
        path = resources.files("aana.tests.files.images") / "Starry_Night.jpeg"
        content = path.read_bytes()
        image = Image(content=content, save_on_disk=False)
        assert image.path is None
        assert image.content == content
        assert image.numpy is None
        assert image.url is None

        numpy = image.get_numpy()
        assert numpy.shape == (720, 909, 3)

        content = image.get_content()
        numpy = load_numpy_from_image_bytes(content)
        assert numpy.shape == (720, 909, 3)
    finally:
        image.cleanup()

    try:
        numpy = np.random.rand(100, 100, 3).astype(np.uint8)
        image = Image(numpy=numpy, save_on_disk=False)
        assert image.path is None
        assert image.content is None
        assert np.array_equal(image.numpy, numpy)
        assert image.url is None

        numpy = image.get_numpy()
        assert np.array_equal(numpy, numpy)

        content = image.get_content()
        numpy = load_numpy_from_image_bytes(content)
        assert np.array_equal(numpy, numpy)
    finally:
        image.cleanup()


def test_image_path_not_exist():
    """Test that the image can't be created from path if the path doesn't exist."""
    path = Path("path/to/image_that_does_not_exist.jpeg")
    with pytest.raises(FileNotFoundError):
        Image(path=path)


def test_save_image(mock_download_file):
    """Test that save_on_disk works."""
    try:
        path = resources.files("aana.tests.files.images") / "Starry_Night.jpeg"
        image = Image(path=path, save_on_disk=True)
        assert image.path == path
        assert image.content is None
        assert image.numpy is None
        assert image.url is None
        assert image.path.exists()
    finally:
        image.cleanup()
    # if image is provided as a path, cleanup() should NOT delete the image
    assert image.path.exists()

    try:
        url = "http://example.com/Starry_Night.jpeg"
        image = Image(url=url, save_on_disk=True)
        assert image.content is None
        assert image.numpy is None
        assert image.url == url
        assert image.path.exists()
    finally:
        image.cleanup()

    try:
        path = resources.files("aana.tests.files.images") / "Starry_Night.jpeg"
        content = path.read_bytes()
        image = Image(content=content, save_on_disk=True)
        assert image.content == content
        assert image.numpy is None
        assert image.url is None
        assert image.path.exists()
    finally:
        image.cleanup()

    try:
        numpy = np.random.rand(100, 100, 3).astype(np.uint8)
        image = Image(numpy=numpy, save_on_disk=True)
        assert image.content is None
        assert np.array_equal(image.numpy, numpy)
        assert image.url is None
        assert image.path.exists()
    finally:
        image.cleanup()


def test_cleanup(mock_download_file):
    """Test that cleanup works."""
    try:
        url = "http://example.com/Starry_Night.jpeg"
        image = Image(url=url, save_on_disk=True)
        assert image.path.exists()
    finally:
        image.cleanup()
        assert not image.path.exists()


def test_at_least_one_input():
    """Test that at least one input is provided."""
    with pytest.raises(ValueError):
        Image(save_on_disk=False)

    with pytest.raises(ValueError):
        Image(save_on_disk=True)


def test_get_pil_image():
    """Test that get_pil_image method."""
    try:
        path = resources.files("aana.tests.files.images") / "Starry_Night.jpeg"
        image = Image(path=path, save_on_disk=False)
        pil_image = image.get_pil_image()
        assert isinstance(pil_image, PIL.Image.Image)
        assert pil_image.size == (909, 720)
    finally:
        image.cleanup()


def test_jpg_format(test_numpy_image):
    """Test that jpg format is used if format is set to jpg."""
    try:
        image = Image(numpy=test_numpy_image, save_on_disk=True, format="jpeg")
        assert image.format == "jpeg"
        assert image.path.suffix == ".jpeg"
    finally:
        image.cleanup()


def test_png_format(test_numpy_image):
    """Test that png format is used if format is set to png."""
    try:
        image = Image(numpy=test_numpy_image, save_on_disk=True, format="png")
        assert image.format == "png"
        assert image.path.suffix == ".png"
    finally:
        image.cleanup()


def test_invalid_format(test_numpy_image):
    """Test that ValueError is raised if an invalid format is provided."""
    with pytest.raises(ValueError):
        Image(numpy=test_numpy_image, save_on_disk=True, format="invalid_format")


def test_get_base64():
    """Test that get_base64 method."""
    try:
        path = resources.files("aana.tests.files.images") / "Starry_Night.jpeg"
        image = Image(path=path, save_on_disk=False)
        base64_content = image.get_base64(format="jpeg")
        img_bytes = base64.b64decode(base64_content, validate=True)
        assert img_bytes.startswith(b"\xff\xd8")  # JPEG magic number
    finally:
        image.cleanup()


def test_get_base64_url():
    """Test that get_base64_url method."""
    try:
        path = resources.files("aana.tests.files.images") / "Starry_Night.jpeg"
        image = Image(path=path, save_on_disk=False)
        base64_url = image.get_base64_url(format="jpeg")
        assert base64_url.startswith("data:image/jpeg;base64,")
        img_bytes = base64.b64decode(base64_url.split(",")[1], validate=True)
        assert img_bytes.startswith(b"\xff\xd8")  # JPEG magic number
    finally:
        image.cleanup()


def test_get_base64_png():
    """Test that get_base64 method with png format."""
    try:
        path = resources.files("aana.tests.files.images") / "Starry_Night.jpeg"
        image = Image(path=path, save_on_disk=False)
        base64_content = image.get_base64(format="png")
        img_bytes = base64.b64decode(base64_content, validate=True)
        assert img_bytes.startswith(b"\x89PNG")  # PNG magic number
    finally:
        image.cleanup()


def test_get_base64_url_from_numpy_as_jpeg(test_numpy_image):
    """Test that get_base64_url method with numpy as jpeg."""
    try:
        image = Image(numpy=test_numpy_image, save_on_disk=False)
        base64_url = image.get_base64_url(format="jpeg")
        assert base64_url.startswith("data:image/jpeg;base64,")
        img_bytes = base64.b64decode(base64_url.split(",")[1], validate=True)
        assert img_bytes.startswith(b"\xff\xd8")  # JPEG magic number
    finally:
        image.cleanup()
