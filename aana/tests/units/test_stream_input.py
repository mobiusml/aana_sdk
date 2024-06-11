# ruff: noqa: S101
import pytest
from pydantic import ValidationError

from aana.core.models.stream import StreamInput


def test_new_stream_input_success():
    """Test that StreamInput can be created successfully."""
    stream_input = StreamInput(url="http://example.com/stream.m3u8")
    assert stream_input.url == "http://example.com/stream.m3u8"


def test_stream_input_invalid_media_id():
    """Test that StreamInput can't be created if media_id is invalid."""
    with pytest.raises(ValidationError):
        StreamInput(url="http://example.com/stream.m3u8", media_id="")


@pytest.mark.parametrize(
    "url, extract_fps",
    [("http://example.com/stream.m3u8", 0), ("http://example.com/stream.m3u8", -1)],
)
def test_stream_input_invalid_extract_fps(url, extract_fps):
    """Test that StreamInput can't be created if extract_fps is invalid."""
    with pytest.raises(ValidationError):
        StreamInput(url=url, extract_fps=extract_fps)


@pytest.mark.parametrize(
    "url, channel_number",
    [("http://example.com/stream.m3u8", -1), ("http://example.com/stream.m3u8", 0.3)],
)
def test_stream_input_invalid_channel(url, channel_number):
    """Test that StreamInput can't be created if channel number is invalid."""
    with pytest.raises(ValidationError):
        StreamInput(url=url, channel_number=channel_number)
