import collections.abc
from types import MappingProxyType

from pydantic import BaseModel, Field, validator


class WhisperParams(BaseModel):
    """A model for the Whisper audio-to-text model parameters.

    Attributes:
        language (str): Optional language code such as "en" or "fr".
                        If None, language will be automatically detected.
        beam_size (int): Size of the beam for decoding.
        best_of (int): Number of best candidate sentences to consider.
        temperature (Union[float, List[float], Tuple[float, ...]]): Controls the sampling
            randomness.  It can be a tuple of temperatures,
            which will be successively used upon failures according to either
            [compression_ratio_threshold](https://github.com/guillaumekln/faster-whisper/blob/5a0541ea7d054aa3716ac492491de30158c20057/faster_whisper/transcribe.py#L216) or
            [log_prob_threshold](https://github.com/guillaumekln/faster-whisper/blob/5a0541ea7d054aa3716ac492491de30158c20057/faster_whisper/transcribe.py#L218C23-L218C23).
        word_timestamps (bool): Whether to extract word-level timestamps.
        vad_filter (bool): Whether to enable voice activity detection to filter non-speech.
    """

    language: str | None = Field(
        default=None,
        description="Language code such as 'en' or 'fr'. If None, language is auto-detected.",
    )
    beam_size: int = Field(
        default=5, ge=1, description="Size of the beam to use for decoding."
    )
    best_of: int = Field(
        default=5, ge=1, description="Number of best candidate sentences to consider."
    )
    temperature: float | collections.abc.Sequence[float] = Field(
        default=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        description=(
            "Temperature for sampling. A single value or a sequence indicating fallback temperatures."
        ),
    )
    word_timestamps: bool = Field(
        default=False, description="Whether to extract word-level timestamps."
    )
    vad_filter: bool = Field(
        default=True,
        description="Whether to enable voice activity detection filtering.",
    )

    @validator("temperature")
    def check_temperature(cls, v: float):
        """Validates a temperature value.

        Args:
            v (float): Value to validate.

        Raises:
            ValueError: Temperature is out of range.

        Returns:
            Temperature value.
        """
        if isinstance(v, float) and not 0 <= v <= 1:
            raise ValueError(  # noqa: TRY003
                "Temperature must be between 0 and 1 when a single float is provided."
            )
        if isinstance(v, list | tuple) and not all(0 <= t <= 1 for t in v):
            raise ValueError(  # noqa: TRY003
                "Each temperature in the sequence must be between 0 and 1."
            )
        return v

    class Config:
        schema_extra = MappingProxyType(
            {"description": "Parameters for the Whisper audio-to-text model."}
        )


class BatchedAsrOptions:
    """A model for the batched version of ASR options.

    Attributes:
    default_batched_asr_options (MappingProxyType): Immutable default options.
    """

    default_batched_asr_options: MappingProxyType = Field(
        default=MappingProxyType(
            {
                "beam_size": 5,
                "best_of": 5,
                "patience": 1,
                "length_penalty": 1,
                "repetition_penalty": 1,
                "no_repeat_ngram_size": 0,
                "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": False,
                "prompt_reset_on_temperature": 0.5,
                "initial_prompt": None,
                "prefix": None,
                "suppress_blank": True,
                "suppress_tokens": [-1],
                "without_timestamps": True,  # False for timings
                "max_initial_timestamp": 0.0,
                "word_timestamps": False,
                "prepend_punctuations": "\"'“¿([{-",
                "append_punctuations": "\"'.。,，!！?？:：”)]}、",  # noqa: RUF001
                "log_prob_low_threshold": -2.0,
                "multilingual": False,
                "output_language": "en",
            },
        ),
        description="default ASR options for the batched ASR model.",
    )


default_batched_asr_options = {
    "beam_size": 5,
    "best_of": 5,
    "patience": 1,
    "length_penalty": 1,
    "repetition_penalty": 1,
    "no_repeat_ngram_size": 0,
    "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "condition_on_previous_text": False,
    "prompt_reset_on_temperature": 0.5,
    "initial_prompt": None,
    "prefix": None,
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": True,  # False for timings
    "max_initial_timestamp": 0.0,
    "word_timestamps": False,
    "prepend_punctuations": "\"'“¿([{-",
    "append_punctuations": "\"'.。,，!！?？:：”)]}、",  # noqa: RUF001
    "log_prob_low_threshold": -2.0,
    "multilingual": False,
    "output_language": "en",
}
