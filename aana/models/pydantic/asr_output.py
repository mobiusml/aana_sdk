from types import MappingProxyType  # for immutable dictionary

import numpy as np
from faster_whisper.transcribe import (
    Segment as WhisperSegment,
)
from faster_whisper.transcribe import (
    TranscriptionInfo as WhisperTranscriptionInfo,
)
from faster_whisper.transcribe import (
    Word as WhisperWord,
)

from aana.models.pydantic.base import BaseListModel
from aana.models.pydantic.time_interval import TimeInterval
from pydantic import BaseModel, Field


class AsrWord(BaseModel):
    """Pydantic schema for Word from ASR model.

    Attributes:
        word (str): The word text
        time_interval (TimeInterval): Time interval of the word
        alignment_confidence (float): Alignment confidence of the word
    """

    word: str = Field(description="The word text")
    time_interval: TimeInterval = Field(description="Time interval of the word")
    alignment_confidence: float = Field(
        ge=0.0, le=1.0, description="Alignment confidence of the word"
    )

    @classmethod
    def from_whisper(cls, whisper_word: WhisperWord) -> "AsrWord":
        """Convert WhisperWord to AsrWord."""
        return cls(
            word=whisper_word.word,
            time_interval=TimeInterval(start=whisper_word.start, end=whisper_word.end),
            alignment_confidence=whisper_word.probability,
        )

    class Config:
        schema_extra = MappingProxyType(
            {
                "description": "Word",
            }
        )


class AsrSegment(BaseModel):
    """Pydantic schema for Segment from ASR model.

    Attributes:
        text (str): The text of the segment (transcript/translation)
        time_interval (TimeInterval): Time interval of the segment
        confidence (float): Confidence of the segment
        no_speech_confidence (float): Chance of being a silence segment
        words (List[AsrWord]): List of words in the segment
    """

    text: str = Field(description="The text of the segment (transcript/translation)")
    time_interval: TimeInterval = Field(description="Time interval of the segment")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence of the segment")
    no_speech_confidence: float = Field(
        ge=0.0, le=1.0, description="Chance of being a silence segment"
    )
    words: list[AsrWord] = Field(
        description="List of words in the segment", default_factory=list
    )

    @classmethod
    def from_whisper(cls, whisper_segment: WhisperSegment) -> "AsrSegment":
        """Convert WhisperSegment to AsrSegment."""
        time_interval = TimeInterval(
            start=whisper_segment.start, end=whisper_segment.end
        )
        confidence = np.exp(whisper_segment.avg_logprob)
        if whisper_segment.words:
            words = [AsrWord.from_whisper(word) for word in whisper_segment.words]
        else:
            words = []

        return cls(
            text=whisper_segment.text,
            time_interval=time_interval,
            confidence=confidence,
            no_speech_confidence=whisper_segment.no_speech_prob,
            words=words,
        )

    class Config:
        schema_extra = MappingProxyType(
            {
                "description": "Segment",
            }
        )


class AsrTranscriptionInfo(BaseModel):
    """Pydantic schema for TranscriptionInfo."""

    language: str = Field(description="Language of the transcription", default="")
    language_confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence of the language detection", default=0.0
    )

    @classmethod
    def from_whisper(
        cls, transcription_info: WhisperTranscriptionInfo
    ) -> "AsrTranscriptionInfo":
        """Convert WhisperTranscriptionInfo to AsrTranscriptionInfo."""
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

    class Config:
        schema_extra = MappingProxyType(
            {
                "description": "Transcription info",
            }
        )


class AsrTranscription(BaseModel):
    """Pydantic schema for Transcription/Translation."""

    text: str = Field(
        description="The text of the transcription/translation", default=""
    )

    def __add__(self, other: "AsrTranscription") -> "AsrTranscription":
        """Sum two transcriptions."""
        text = self.text + "\n" + other.text
        return AsrTranscription(text=text)

    class Config:
        schema_extra = MappingProxyType(
            {
                "description": "Transcription/Translation",
            }
        )


class AsrSegments(BaseListModel):
    """Pydantic schema for the list of ASR segments."""

    __root__: list[AsrSegment] = Field(
        description="List of ASR segments", default_factory=list
    )

    class Config:
        schema_extra = MappingProxyType(
            {
                "description": "List of ASR segments",
            }
        )


class AsrSegmentsList(BaseListModel):
    """Pydantic schema for the list of lists of ASR segments."""

    __root__: list[AsrSegments]

    class Config:
        schema_extra = MappingProxyType(
            {
                "description": "List of lists of ASR segments",
            }
        )


class AsrTranscriptionInfoList(BaseListModel):
    """Pydantic schema for the list of ASR transcription info."""

    __root__: list[AsrTranscriptionInfo]

    class Config:
        schema_extra = MappingProxyType(
            {
                "description": "List of ASR transcription info",
            }
        )


class AsrTranscriptionList(BaseListModel):
    """Pydantic schema for the list of ASR transcription."""

    __root__: list[AsrTranscription]

    class Config:
        schema_extra = MappingProxyType(
            {
                "description": "List of ASR transcription",
            }
        )
