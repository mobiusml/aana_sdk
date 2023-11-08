from typing import List, Optional
from pydantic import BaseModel, Field
from faster_whisper.transcribe import (
    Segment as WhisperSegment,
    Word as WhisperWord,
)


class AsrWord(BaseModel):
    """
    Pydantic schema for Word from ASR model.
    """

    start: float = Field(description="Start time of the word")
    end: float = Field(description="End time of the word")
    word: str = Field(description="The word text")
    probability: float = Field(description="Probability confidence of the word")

    @classmethod
    def from_namedtuple(cls, namedtuple_word: WhisperWord) -> "AsrWord":
        return cls(**namedtuple_word._asdict())


class AsrSegment(BaseModel):
    """
    Pydantic schema for Segment from ASR model.
    """

    start: float = Field(description="Start time of the segment")
    end: float = Field(description="End time of the segment")
    text: str = Field(description="Transcription text of the segment")
    no_speech_prob: float = Field(description="Probability of no speech in the segment")
    words: Optional[List[AsrWord]] = Field(
        description="List of words in the segment", default=None
    )

    @classmethod
    def from_namedtuple(cls, namedtuple_segment: WhisperSegment) -> "AsrSegment":
        segment_data = namedtuple_segment._asdict()
        segment_data["words"] = (
            [AsrWord.from_namedtuple(word) for word in segment_data["words"]]
            if segment_data["words"]
            else None
        )
        return cls(**segment_data)
