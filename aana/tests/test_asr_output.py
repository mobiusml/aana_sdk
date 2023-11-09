import numpy as np
import pytest
from faster_whisper.transcribe import (
    Segment as WhisperSegment,
    Word as WhisperWord,
    TranscriptionInfo as WhisperTranscriptionInfo,
)
from aana.models.pydantic.asr_output import (
    AsrSegment,
    AsrTranscriptionInfo,
    AsrWord,
    Timestamp,
)


def test_asr_segment_from_whisper():
    """
    Test function for the AsrSegment class's from_whisper method.
    """
    whisper_segment = WhisperSegment(
        id=0,
        seek=0,
        tokens=[],
        temperature=0.0,
        compression_ratio=0.0,
        start=0.0,
        end=1.0,
        avg_logprob=-0.5,
        no_speech_prob=0.1,
        words=[],
        text="hello world",
    )

    asr_segment = AsrSegment.from_whisper(whisper_segment)

    assert asr_segment.text == "hello world"
    assert asr_segment.timestamp == Timestamp(
        start=whisper_segment.start, end=whisper_segment.end
    )
    assert asr_segment.confidence == np.exp(whisper_segment.avg_logprob)
    assert asr_segment.no_speech_confidence == whisper_segment.no_speech_prob
    assert asr_segment.words == []

    word = WhisperWord(
        word="hello",
        start=0.0,
        end=0.5,
        probability=0.5,
    )
    whisper_segment = WhisperSegment(
        id=0,
        seek=0,
        tokens=[],
        temperature=0.0,
        compression_ratio=0.0,
        start=0.0,
        end=1.0,
        avg_logprob=-0.5,
        no_speech_prob=0.1,
        words=[word],
        text="hello world",
    )

    asr_segment = AsrSegment.from_whisper(whisper_segment)
    assert asr_segment.words == [AsrWord.from_whisper(word)]


def test_asr_word_from_whisper():
    """
    Test function for the AsrWord class's from_whisper method.
    """
    word = WhisperWord(
        word="hello",
        start=0.0,
        end=0.5,
        probability=0.5,
    )

    asr_word = AsrWord.from_whisper(word)

    assert asr_word.word == "hello"
    assert asr_word.timestamp == Timestamp(start=word.start, end=word.end)
    assert asr_word.alignment_confidence == word.probability
