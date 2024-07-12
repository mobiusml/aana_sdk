# ruff: noqa: S101, NPY002
import io
from importlib import resources
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from aana.core.models.image import ImageInput, ImageInputList


@pytest.fixture
def mock_download_file(mocker):
    """Mock download_file."""
    mock = mocker.patch("aana.core.models.image.download_file", autospec=True)
    path = resources.path("aana.tests.files.images", "Starry_Night.jpeg")
    content = path.read_bytes()
    mock.return_value = content
    return mock


def test_new_imageinput_success():
    """Test that ImageInput can be created successfully."""
    image_input = ImageInput(path="image.png")
    assert image_input.path == "image.png"

    image_input = ImageInput(url="http://image.png")
    assert image_input.url == "http://image.png/"

    image_input = ImageInput(content=b"file")
    assert image_input.content == b"file"

    image_input = ImageInput(numpy=b"file")
    assert image_input.numpy == b"file"


def test_imageinput_invalid_media_id():
    """Test that ImageInput can't be created if media_id is invalid."""
    with pytest.raises(ValidationError):
        ImageInput(path="image.png", media_id="")


@pytest.mark.parametrize(
    "url",
    [
        "domain",
        "domain.com",
        "http://",
        "www.domain.com",
        "/subdomain",
        "../subdomain",
        "",
    ],
)
def test_imageinput_invalid_url(url):
    """Test that ImageInput can't be created if url is invalid."""
    with pytest.raises(ValidationError):
        ImageInput(url=url)


def test_imageinput_check_only_one_field():
    """Test that exactly one of 'path', 'url', 'content', or 'numpy' is provided."""
    fields = {
        "path": "image.png",
        "url": "http://image.png",
        "content": b"file",
        "numpy": b"file",
    }

    # check all combinations of two fields
    for field1 in fields:
        for field2 in fields:
            if field1 != field2:
                with pytest.raises(ValidationError):
                    ImageInput(**{field1: fields[field1], field2: fields[field2]})

    # check all combinations of three fields
    for field1 in fields:
        for field2 in fields:
            for field3 in fields:
                if field1 != field2 and field1 != field3 and field2 != field3:
                    with pytest.raises(ValidationError):
                        ImageInput(
                            **{
                                field1: fields[field1],
                                field2: fields[field2],
                                field3: fields[field3],
                            }
                        )

    # check all combinations of four fields
    with pytest.raises(ValidationError):
        ImageInput(**fields)

    # check that no fields is also invalid
    with pytest.raises(ValidationError):
        ImageInput()


def test_imageinput_set_file():
    """Test that the file can be set for the image."""
    file_content = b"image data"

    # If 'content' is set to 'file',
    # the image can be set from the file uploaded to the endpoint.
    image_input = ImageInput(content=b"file")
    image_input.set_file(file_content)
    assert image_input.content == file_content

    # If 'numpy' is set to 'file',
    # the image can be set from the file uploaded to the endpoint.
    image_input = ImageInput(numpy=b"file")
    image_input.set_file(file_content)
    assert image_input.numpy == file_content

    # If neither 'content' nor 'numpy' is set to 'file',
    # an error should be raised.
    image_input = ImageInput(path="image.png")
    with pytest.raises(ValueError):
        image_input.set_file(file_content)


def test_imageinput_set_files():
    """Test that the files can be set for the image."""
    files = [b"image data"]

    # If 'content' is set to 'file',
    # the image can be set from the file uploaded to the endpoint.
    image_input = ImageInput(content=b"file")
    image_input.set_files(files)
    assert image_input.content == files[0]

    # If 'numpy' is set to 'file',
    # the image can be set from the file uploaded to the endpoint.
    image_input = ImageInput(numpy=b"file")
    image_input.set_files(files)
    assert image_input.numpy == files[0]

    # If neither 'content' nor 'numpy' is set to 'file',
    # an error should be raised.
    image_input = ImageInput(path="image.png")
    with pytest.raises(ValueError):
        image_input.set_files(files)

    # If the number of images and files aren't the same,
    # an error should be raised.
    files = [b"image data", b"another image data"]
    image_input = ImageInput(content=b"file")
    with pytest.raises(ValidationError):
        image_input.set_files(files)

    files = []
    image_input = ImageInput(content=b"file")
    with pytest.raises(ValidationError):
        image_input.set_files(files)


def test_imageinput_convert_input_to_object(mock_download_file):
    """Test that ImageInput can be converted to Image."""
    path = resources.path("aana.tests.files.images", "Starry_Night.jpeg")
    image_input = ImageInput(path=str(path))
    try:
        image_object = image_input.convert_input_to_object()
        assert image_object.path == path
    finally:
        image_object.cleanup()

    url = "http://example.com/Starry_Night.jpeg"
    image_input = ImageInput(url=url)
    try:
        image_object = image_input.convert_input_to_object()
        assert image_object.url == url
    finally:
        image_object.cleanup()

    content = Path(path).read_bytes()
    image_input = ImageInput(content=content)
    try:
        image_object = image_input.convert_input_to_object()
        assert image_object.content == content
    finally:
        image_object.cleanup()

    numpy = np.random.rand(100, 100, 3).astype(np.uint8)
    # convert numpy array to bytes
    buffer = io.BytesIO()
    np.save(buffer, numpy)
    numpy_bytes = buffer.getvalue()
    image_input = ImageInput(numpy=numpy_bytes)
    try:
        image_object = image_input.convert_input_to_object()
        assert np.array_equal(image_object.numpy, numpy)
    finally:
        image_object.cleanup()


def test_imageinput_convert_input_to_object_invalid_numpy():
    """Test that ImageInput can't be converted to Image if numpy is invalid."""
    numpy = np.random.rand(100, 100, 3).astype(np.uint8)
    # convert numpy array to bytes
    buffer = io.BytesIO()
    np.save(buffer, numpy)
    numpy_bytes = buffer.getvalue()
    # remove the last byte
    numpy_bytes = numpy_bytes[:-1]
    image_input = ImageInput(numpy=numpy_bytes)
    with pytest.raises(ValueError):
        image_input.convert_input_to_object()


def test_imageinput_convert_input_to_object_numpy_not_set():
    """Test that ImageInput can't be converted to Image if numpy file isn't set with set_file()."""
    image_input = ImageInput(numpy=b"file")
    with pytest.raises(ValueError):
        image_input.convert_input_to_object()


def test_imagelistinput():
    """Test that ImageListInput can be created successfully."""
    images = [
        ImageInput(path="image.png"),
        ImageInput(url="http://image.png"),
        ImageInput(content=b"file"),
        ImageInput(numpy=b"file"),
    ]

    image_list_input = ImageInputList(images)
    assert image_list_input.root == images
    assert len(image_list_input) == len(images)
    assert image_list_input[0] == images[0]
    assert image_list_input[1] == images[1]
    assert image_list_input[2] == images[2]
    assert image_list_input[3] == images[3]


def test_imagelistinput_set_files():
    """Test that the files can be set for the images."""
    files = [b"image data 1", b"image data 2"]

    images = [
        ImageInput(content=b"file"),
        ImageInput(numpy=b"file"),
    ]

    image_list_input = ImageInputList(images)
    image_list_input.set_files(files)

    assert image_list_input[0].content == files[0]
    assert image_list_input[1].numpy == files[1]


def test_imagelistinput_non_empty():
    """Test that ImageInputList must not be empty."""
    with pytest.raises(ValidationError):
        ImageInputList([])
