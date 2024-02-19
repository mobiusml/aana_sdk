# ruff: noqa: S101
import pytest
from pydantic import ValidationError

from aana.models.pydantic.media_id import MediaId


def test_media_id_creation():
    """Test that a media id can be created."""
    media_id = MediaId(__root__="foo")
    assert media_id == "foo"

    media_id = MediaId("foo")
    assert media_id == "foo"

    with pytest.raises(ValueError):
        media_id = MediaId()

    with pytest.raises(ValidationError):
        media_id = MediaId("")


def test_media_id_random():
    """Test that a random media id can be created."""
    media_id = MediaId.random()
    assert media_id is not None
    assert media_id != ""
