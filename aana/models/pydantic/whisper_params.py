from pydantic import BaseModel, Field, validator
from typing import Optional, Union, List, Tuple


class WhisperParams(BaseModel):
    """
    A model for the Whisper audio-to-text model parameters.

    Attributes:
        language (str): Optional language code such as "en" or "fr".
                        If None, language will be automatically detected.
        beam_size (int): Size of the beam for decoding.
        best_of (int): Number of best candidate sentences to consider.
        temperature (Union[float, List[float], Tuple[float, ...]]): Controls the sampling
            randomness, with a sequence of values indicating fallback temperatures.
        word_timestamps (bool): Whether to extract word-level timestamps.
        vad_filter (bool): Whether to enable voice activity detection to filter non-speech.
    """

    language: Optional[str] = Field(
        default=None,
        description="Language code such as 'en' or 'fr'. If None, language is auto-detected.",
    )
    beam_size: int = Field(
        default=5, ge=1, description="Size of the beam to use for decoding."
    )
    best_of: int = Field(
        default=5, ge=1, description="Number of best candidate sentences to consider."
    )
    temperature: Union[float, List[float], Tuple[float, ...]] = Field(
        default=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        description=(
            "Temperature for sampling. A single value or a sequence indicating fallback temperatures."
        ),
    )
    word_timestamps: bool = Field(
        default=False, description="Whether to extract word-level timestamps."
    )
    vad_filter: bool = Field(
        default=False,
        description="Whether to enable voice activity detection filtering.",
    )

    @validator("temperature")
    def check_temperature(cls, v):
        if isinstance(v, float) and not 0 <= v <= 1:
            raise ValueError(
                "Temperature must be between 0 and 1 when a single float is provided."
            )
        if isinstance(v, (list, tuple)) and not all(0 <= t <= 1 for t in v):
            raise ValueError(
                "Each temperature in the sequence must be between 0 and 1."
            )
        return v

    class Config:
        schema_extra = {
            "description": "Parameters for the Whisper audio-to-text model."
        }