from importlib import resources
from pathlib import Path
import numpy as np
import pytest
from aana.models.core.image import Image


def load_numpy_from_image_bytes(content: bytes) -> np.ndarray:
    """
    Load a numpy array from image bytes.
    """
    image = Image(content=content, save_on_disk=False)
    return image.get_numpy()


@pytest.fixture
def mock_download_file(mocker):
    """
    Mock download_file.
    """
    mock = mocker.patch("aana.models.core.image.download_file", autospec=True)
    path = resources.path("aana.tests.files.images", "Starry_Night.jpeg")
    content = path.read_bytes()
    mock.return_value = content
    return mock


def test_image(mock_download_file):
    """
    Test that the image can be created from path, url, content, or numpy.
    """

    try:
        path = resources.path("aana.tests.files.images", "Starry_Night.jpeg")
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
        path = resources.path("aana.tests.files.images", "Starry_Night.jpeg")
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
    """
    Test that the image can't be created from path if the path doesn't exist.
    """
    path = Path("path/to/image_that_does_not_exist.jpeg")
    with pytest.raises(FileNotFoundError):
        Image(path=path)


def test_save_image(mock_download_file):
    """
    Test that save_on_disc works.
    """

    try:
        path = resources.path("aana.tests.files.images", "Starry_Night.jpeg")
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
        path = resources.path("aana.tests.files.images", "Starry_Night.jpeg")
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
    """
    Test that cleanup works.
    """

    try:
        url = "http://example.com/Starry_Night.jpeg"
        image = Image(url=url, save_on_disk=True)
        assert image.path.exists()
    finally:
        image.cleanup()
        assert not image.path.exists()


def test_at_least_one_input():
    """
    Test that at least one input is provided.
    """
    with pytest.raises(ValueError):
        Image(save_on_disk=False)

    with pytest.raises(ValueError):
        Image(save_on_disk=True)
