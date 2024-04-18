import collections.abc

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MyConfig(ConfigDict, total=False):  # noqa: D101
    json_schema_extra: dict


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

    @field_validator("temperature")
    @classmethod
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

    model_config = MyConfig(
        json_schema_extra={
            "description": "Parameters for the Whisper audio-to-text model."
        }
    )


class BatchedWhisperParams(BaseModel):
    """A model for the Batched version of Whisper audio-to-text model parameters.

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
        #TODO: add other parameters
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

    @field_validator("temperature")
    @classmethod
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

    model_config = MyConfig(
        json_schema_extra={
            "description": "Parameters for the Batched Whisper audio-to-text model."
        }
    )
