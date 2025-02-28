# ruff: noqa: S101

import pytest
from pydantic import BaseModel, ValidationError

from aana.core.models.media import MediaId


class TestModel(BaseModel):
    """Test model for media id."""

    media_id: MediaId


def test_media_id_creation():
    """Test that a media id can be created."""
    media_id = TestModel(media_id="foo").media_id
    assert media_id == "foo"

    # Validation only happens when the model is created
    # because MediaId is just an annotated string
    with pytest.raises(ValueError):
        TestModel().media_id  # noqa: B018

    with pytest.raises(ValidationError):
        TestModel(media_id="").media_id  # noqa: B018

    # MediaId is a string with a maximum length of 36
    with pytest.raises(ValidationError):
        TestModel(media_id="a" * 37).media_id  # noqa: B018


def test_valid_media_ids():
    """Test that valid media ids are accepted."""
    valid_ids = ["abc123", "abc-123", "abc_123", "A1B2_C3", "123456", "a_b-c"]
    for media_id in valid_ids:
        model = TestModel(media_id=media_id)
        assert model.media_id == media_id


def test_invalid_media_ids():
    """Test that invalid media ids are rejected."""
    invalid_ids = [
        "abc 123",  # contains a space
        "abc@123",  # contains an invalid character (@)
        "abc#123",  # contains an invalid character (#)
        "abc.123",  # contains an invalid character (.)
    ]
    for media_id in invalid_ids:
        with pytest.raises(ValidationError):
            TestModel(media_id=media_id)
