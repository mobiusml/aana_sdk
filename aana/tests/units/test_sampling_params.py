# ruff: noqa: S101
import pytest

from aana.core.models.sampling import SamplingParams


def test_valid_sampling_params():
    """Test valid sampling parameters."""
    params = SamplingParams(temperature=0.5, top_p=0.9, top_k=10, max_tokens=50)
    assert params.temperature == 0.5
    assert params.top_p == 0.9
    assert params.top_k == 10
    assert params.max_tokens == 50

    # Test valid params with default values (None)
    params = SamplingParams()
    assert params.temperature is None
    assert params.top_p is None
    assert params.top_k is None
    assert params.max_tokens is None


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
