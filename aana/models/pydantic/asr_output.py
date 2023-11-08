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
    """

    text: str = Field(description="The text of the segment (transcript/translation)")
    timestamp: Timestamp = Field(description="Timestamp of the segment")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence of the segment")
    no_speech_confidence: float = Field(
        ge=0.0, le=1.0, description="Chance of being a silence segment"
    )
    words: Optional[List[AsrWord]] = Field(
        description="List of words in the segment", default=None
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
            words = None

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


class AsrOutput(BaseModel):
    """
    Pydantic schema for ASR output.
    """

    segments: List[AsrSegment] = Field(description="List of segments")
    transcription_info: AsrTranscriptionInfo = Field(description="Transcription info")

    @classmethod
    def from_whisper(
        cls,
        segments: List[WhisperSegment],
        transcription_info: WhisperTranscriptionInfo,
    ) -> "AsrOutput":
        """
        Convert Whisper output to ASR output.
        """
        return cls(
            segments=[AsrSegment.from_whisper(seg) for seg in segments],
            transcription_info=AsrTranscriptionInfo.from_whisper(transcription_info),
        )

    class Config:
        schema_extra = {
            "description": "ASR output",
        }


class AsrOutputList(BaseListModel):
    """
    Pydantic schema for the list of ASR outputs.
    """

    __root__: List[AsrOutput]

    class Config:
        schema_extra = {
            "description": "List of ASR outputs",
        }
