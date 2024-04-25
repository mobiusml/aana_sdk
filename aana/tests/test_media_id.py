# ruff: noqa: S101
import uuid

import pytest
from pydantic import BaseModel, ValidationError

from aana.models.pydantic.media_id import MediaId


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
