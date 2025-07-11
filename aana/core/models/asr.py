import typing
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from aana.core.models.time import TimeInterval

if typing.TYPE_CHECKING:
    from faster_whisper.transcribe import (
        Segment as WhisperSegment,
    )
    from faster_whisper.transcribe import (
        TranscriptionInfo as WhisperTranscriptionInfo,
    )
    from faster_whisper.transcribe import (
        Word as WhisperWord,
    )

__all__ = [
    "AsrSegment",
    "AsrSegments",
    "AsrSegmentsList",
    "AsrTranscription",
    "AsrTranscriptionInfo",
    "AsrTranscriptionInfoList",
    "AsrTranscriptionList",
    "AsrWord",
]


class AsrWord(BaseModel):
    """Pydantic schema for Word from ASR model.

    Attributes:
        word (str): The word text.
        speaker (str| None): Speaker label for the word.
        time_interval (TimeInterval): Time interval of the word.
        alignment_confidence (float): Alignment confidence of the word, >= 0.0 and <= 1.0.
    """

    word: str = Field(description="The word text")
    speaker: str | None = Field(None, description="Speaker label for the word")
    time_interval: TimeInterval = Field(description="Time interval of the word")
    alignment_confidence: float = Field(
        ge=0.0, le=1.0, description="Alignment confidence of the word"
    )

    @classmethod
    def from_whisper(cls, whisper_word: "WhisperWord") -> "AsrWord":
        """Convert WhisperWord to AsrWord.

        Args:
            whisper_word (WhisperWord): The WhisperWord from faster-whisper.

        Returns:
            AsrWord: The converted AsrWord.
        """
        return cls(
            speaker=None,
            word=whisper_word.word,
            time_interval=TimeInterval(start=whisper_word.start, end=whisper_word.end),
            alignment_confidence=whisper_word.probability,
        )

    model_config = ConfigDict(
        json_schema_extra={"description": "Word"},
        extra="forbid",
    )


class AsrSegment(BaseModel):
    """Pydantic schema for Segment from ASR model.

    Attributes:
        text (str): The text of the segment (transcript/translation).
        time_interval (TimeInterval): Time interval of the segment.
        confidence (float | None): Confidence of the segment.
        no_speech_confidence (float | None): Chance of being a silence segment.
        words (list[AsrWord]): List of words in the segment. Default is [].
        speaker (str | None): Speaker label. Default is None.
    """

    text: str = Field(description="The text of the segment (transcript/translation)")
    time_interval: TimeInterval = Field(description="Time interval of the segment")
    confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Confidence of the segment"
    )
    no_speech_confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Chance of being a silence segment"
    )
    words: list[AsrWord] = Field(
        description="List of words in the segment", default_factory=list
    )
    speaker: str | None = Field(None, description="speaker label of the segment")

    @classmethod
    def from_whisper(cls, whisper_segment: "WhisperSegment") -> "AsrSegment":
        """Convert WhisperSegment to AsrSegment."""
        time_interval = TimeInterval(
            start=whisper_segment.start, end=whisper_segment.end
        )
        try:
            avg_logprob = whisper_segment.avg_logprob
            confidence = np.exp(avg_logprob)
        except AttributeError:
            confidence = None

        try:
            words = [AsrWord.from_whisper(word) for word in whisper_segment.words]
        except TypeError:  # "None type object is not iterable"
            words = []
        except AttributeError:  # "'StreamSegment' object has no attribute 'words'"
            words = []
        try:
            no_speech_confidence = whisper_segment.no_speech_prob
        except AttributeError:
            no_speech_confidence = None

        return cls(
            text=whisper_segment.text,
            time_interval=time_interval,
            confidence=confidence,
            no_speech_confidence=no_speech_confidence,
            words=words,
            speaker=None,
        )

    model_config = ConfigDict(
        json_schema_extra={"description": "Segment"},
        extra="forbid",
    )


class AsrTranscriptionInfo(BaseModel):
    """Pydantic schema for TranscriptionInfo.

    Attributes:
        language (str): Language of the transcription.
        language_confidence (float): Confidence of the language detection, >= 0.0 and <= 1.0. Default is 0.0.
    """

    language: str = Field(description="Language of the transcription", default="")
    language_confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence of the language detection", default=0.0
    )

    @classmethod
    def from_whisper(
        cls, transcription_info: "WhisperTranscriptionInfo"
    ) -> "AsrTranscriptionInfo":
        """Convert WhisperTranscriptionInfo to AsrTranscriptionInfo.

        Args:
            transcription_info (WhisperTranscriptionInfo): The WhisperTranscriptionInfo from faster-whisper.

        Returns:
            AsrTranscriptionInfo: The converted AsrTranscriptionInfo.
        """
        return cls(
            language=transcription_info.language,
            language_confidence=transcription_info.language_probability,
        )

    def __add__(self, other: "AsrTranscriptionInfo") -> "AsrTranscriptionInfo":
        """Sum two transcriptions info."""
        if self.language == other.language:
            # if the languages are the same, take the average of the confidence
            language = self.language
            language_confidence = (
                self.language_confidence + other.language_confidence
            ) / 2
        else:
            # if the languages are different, take the one with the highest confidence
            if self.language_confidence > other.language_confidence:
                language = self.language
                language_confidence = self.language_confidence
            else:
                language = other.language
                language_confidence = other.language_confidence
        return AsrTranscriptionInfo(
            language=language, language_confidence=language_confidence
        )

    model_config = ConfigDict(
        json_schema_extra={"description": "Transcription info"},
        extra="forbid",
    )


class AsrTranscription(BaseModel):
    """Pydantic schema for Transcription/Translation.

    Attributes:
        text (str): The text of the transcription/translation. Default is "".
    """

    text: str = Field(
        description="The text of the transcription/translation", default=""
    )

    def __add__(self, other: "AsrTranscription") -> "AsrTranscription":
        """Sum two transcriptions."""
        if self.text == "":
            text = other.text
        elif other.text == "":
            text = self.text
        else:
            text = self.text + "\n" + other.text
        return AsrTranscription(text=text)

    model_config = ConfigDict(
        json_schema_extra={"description": "Transcription/Translation"},
        extra="forbid",
    )


AsrSegments = Annotated[
    list[AsrSegment], Field(..., description="List of ASR segments")
]

AsrSegmentsList = Annotated[
    list[AsrSegments], Field(..., description="List of lists of ASR segments")
]
AsrTranscriptionInfoList = Annotated[
    list[AsrTranscriptionInfo], Field(..., description="List of ASR transcription info")
]
AsrTranscriptionList = Annotated[
    list[AsrTranscription], Field(..., description="List of ASR transcription")
]
