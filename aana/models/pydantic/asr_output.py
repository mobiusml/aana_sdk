from typing import List, Optional
import numpy as np
from pydantic import BaseModel, Field
from faster_whisper.transcribe import (
    Segment as WhisperSegment,
    Word as WhisperWord,
    TranscriptionInfo as WhisperTranscriptionInfo,
)

from aana.models.pydantic.base import BaseListModel


class Timestamp(BaseModel):
    """
    Pydantic schema for Timestamp.

    Attributes:
        start (float): Start time
        end (float): End time
    """

    start: float = Field(ge=0.0, description="Start time")
    end: float = Field(ge=0.0, description="End time")

    class Config:
        schema_extra = {
            "description": "Timestamp",
        }


class AsrWord(BaseModel):
    """
    Pydantic schema for Word from ASR model.

    Attributes:
        word (str): The word text
        timestamp (Timestamp): Timestamp of the word
        alignment_confidence (float): Alignment confidence of the word
    """

    word: str = Field(description="The word text")
    timestamp: Timestamp = Field(description="Timestamp of the word")
    alignment_confidence: float = Field(
        ge=0.0, le=1.0, description="Alignment confidence of the word"
    )

    @classmethod
    def from_whisper(cls, whisper_word: WhisperWord) -> "AsrWord":
        """
        Convert WhisperWord to AsrWord.
        """
        return cls(
            word=whisper_word.word,
            timestamp=Timestamp(start=whisper_word.start, end=whisper_word.end),
            alignment_confidence=whisper_word.probability,
        )

    class Config:
        schema_extra = {
            "description": "Word",
        }


class AsrSegment(BaseModel):
    """
    Pydantic schema for Segment from ASR model.

    Attributes:
        text (str): The text of the segment (transcript/translation)
        timestamp (Timestamp): Timestamp of the segment
        confidence (float): Confidence of the segment
        no_speech_confidence (float): Chance of being a silence segment
        words (Optional[List[AsrWord]]): List of words in the segment
    """

    text: str = Field(description="The text of the segment (transcript/translation)")
    timestamp: Timestamp = Field(description="Timestamp of the segment")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence of the segment")
    no_speech_confidence: float = Field(
        ge=0.0, le=1.0, description="Chance of being a silence segment"
    )
    words: Optional[List[AsrWord]] = Field(
        description="List of words in the segment", default_factory=list
    )

    @classmethod
    def from_whisper(cls, whisper_segment: WhisperSegment) -> "AsrSegment":
        """
        Convert WhisperSegment to AsrSegment.
        """
        timestamp = Timestamp(start=whisper_segment.start, end=whisper_segment.end)
        confidence = np.exp(whisper_segment.avg_logprob)
        if whisper_segment.words:
            words = [AsrWord.from_whisper(word) for word in whisper_segment.words]
        else:
            words = []

        return cls(
            text=whisper_segment.text,
            timestamp=timestamp,
            confidence=confidence,
            no_speech_confidence=whisper_segment.no_speech_prob,
            words=words,
        )

    class Config:
        schema_extra = {
            "description": "Segment",
        }


class AsrTranscriptionInfo(BaseModel):
    """
    Pydantic schema for TranscriptionInfo.
    """

    language: str = Field(description="Language of the transcription")
    language_confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence of the language detection"
    )

    @classmethod
    def from_whisper(
        cls, transcription_info: WhisperTranscriptionInfo
    ) -> "AsrTranscriptionInfo":
        """
        Convert WhisperTranscriptionInfo to AsrTranscriptionInfo.
        """
        return cls(
            language=transcription_info.language,
            language_confidence=transcription_info.language_probability,
        )

    class Config:
        schema_extra = {
            "description": "Transcription info",
        }


class AsrTranscription(BaseModel):
    """
    Pydantic schema for Transcription/Translation.
    """

    text: str = Field(description="The text of the transcription/translation")

    class Config:
        schema_extra = {
            "description": "Transcription/Translation",
        }


class AsrSegments(BaseListModel):
    """
    Pydantic schema for the list of ASR segments.
    """

    __root__: List[AsrSegment]

    class Config:
        schema_extra = {
            "description": "List of ASR segments",
        }


class AsrSegmentsList(BaseListModel):
    """
    Pydantic schema for the list of lists of ASR segments.
    """

    __root__: List[AsrSegments]

    class Config:
        schema_extra = {
            "description": "List of lists of ASR segments",
        }


class AsrTranscriptionInfoList(BaseListModel):
    """
    Pydantic schema for the list of ASR transcription info.
    """

    __root__: List[AsrTranscriptionInfo]

    class Config:
        schema_extra = {
            "description": "List of ASR transcription info",
        }


class AsrTranscriptionList(BaseListModel):
    """
    Pydantic schema for the list of ASR transcription.
    """

    __root__: List[AsrTranscription]

    class Config:
        schema_extra = {
            "description": "List of ASR transcription",
        }
