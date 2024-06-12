# ruff: noqa: S101

import pytest
from pydantic import BaseModel, ValidationError

from aana.core.models.media import MediaId


def test_media_id_creation():
    """Test that a media id can be created."""

    class TestModel(BaseModel):
        media_id: MediaId

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
