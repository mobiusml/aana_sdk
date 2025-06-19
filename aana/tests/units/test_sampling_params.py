# ruff: noqa: S101
import pytest
from pydantic import ValidationError

from aana.core.models.sampling import SamplingParams


def test_valid_sampling_params():
    """Test valid sampling parameters."""
    params = SamplingParams(
        temperature=0.5, top_p=0.9, top_k=10, max_tokens=50, repetition_penalty=1.5
    )
    assert params.temperature == 0.5
    assert params.top_p == 0.9
    assert params.top_k == 10
    assert params.max_tokens == 50
    assert params.repetition_penalty == 1.5

    # Test valid params with default values
    params = SamplingParams()
    assert params.temperature == 1.0
    assert params.top_p == 1.0
    assert params.top_k is None
    assert params.max_tokens is None
    assert params.repetition_penalty == 1.0


def test_invalid_temperature():
    """Test invalid temperature values."""
    with pytest.raises(ValueError):
        SamplingParams(temperature=-1.0)


def test_invalid_top_p():
    """Test invalid top_p values."""
    with pytest.raises(ValueError):
        SamplingParams(top_p=0.0)
    with pytest.raises(ValueError):
        SamplingParams(top_p=1.1)


def test_invalid_top_k():
    """Test invalid top_k values."""
    with pytest.raises(ValueError):
        SamplingParams(top_k=0)
    with pytest.raises(ValueError):
        SamplingParams(top_k=-2)


def test_invalid_max_tokens():
    """Test invalid max_tokens values."""
    with pytest.raises(ValueError):
        SamplingParams(max_tokens=0)


def test_kwargs():
    """Test extra keyword arguments."""
    params = SamplingParams(
        temperature=0.5, kwargs={"presence_penalty": 2.0, "frequency_penalty": 1.0}
    )
    assert params.kwargs == {"presence_penalty": 2.0, "frequency_penalty": 1.0}
    assert params.temperature == 0.5
    assert params.top_p == 1.0
    assert params.top_k is None
    assert params.max_tokens is None
    assert params.repetition_penalty == 1.0


def test_disallowed_extra_fields():
    """Test that extra fields are not allowed."""
    with pytest.raises(ValidationError):
        SamplingParams(temperature=0.5, extra_field="extra_value")


def test_guided_decoding_backend_with_options():
    """Test guided decoding backend with additional options."""
    params = SamplingParams(guided_decoding_backend="xgrammar:no-fallback")
    assert params.guided_decoding_backend == "xgrammar:no-fallback"

    with pytest.raises(ValidationError):
        SamplingParams(guided_decoding_backend="invalid-backend")
