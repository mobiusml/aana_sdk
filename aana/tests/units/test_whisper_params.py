# ruff: noqa: S101
import pytest

from aana.core.models.whisper import WhisperParams


def test_whisper_params_default():
    """Test the default values of WhisperParams object.

    Keeping the default parameters of a function or object is important
    in case other code relies on them.

    If you need to change the default parameters, think twice before doing so.
    """
    params = WhisperParams()

    assert params.language is None
    assert params.beam_size == 5
    assert params.best_of == 5
    assert params.temperature == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    assert params.word_timestamps is False
    assert params.vad_filter is True


@pytest.mark.parametrize(
    "language, beam_size, best_of, temperature, word_timestamps, vad_filter",
    [
        ("en", 5, 5, 0.5, True, True),
        ("fr", 3, 3, 0.2, False, False),
        (None, 1, 1, [0.8, 0.9], True, False),
    ],
)
def test_whisper_params(
    language, beam_size, best_of, temperature, word_timestamps, vad_filter
):
    """Test function for the WhisperParams class with valid parameters."""
    params = WhisperParams(
        language=language,
        beam_size=beam_size,
        best_of=best_of,
        temperature=temperature,
        word_timestamps=word_timestamps,
        vad_filter=vad_filter,
    )

    assert params.language == language
    assert params.beam_size == beam_size
    assert params.best_of == best_of
    assert params.temperature == temperature
    assert params.word_timestamps == word_timestamps
    assert params.vad_filter == vad_filter


@pytest.mark.parametrize(
    "temperature",
    [
        [-1.0, 0.5, 1.5],
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0],
        "invalid_temperature",
        2,
    ],
)
def test_whisper_params_invalid_temperature(temperature):
    """Check ValueError raised if temperature is invalid."""
    with pytest.raises(ValueError):
        WhisperParams(temperature=temperature)
