# ruff: noqa: S101
import numpy as np
from faster_whisper.transcribe import (
    Segment as WhisperSegment,
)
from faster_whisper.transcribe import (
    Word as WhisperWord,
)

from aana.core.models.asr import (
    AsrSegment,
    AsrSegments,
    AsrTranscription,
    AsrTranscriptionInfo,
    AsrWord,
)
from aana.core.models.time import TimeInterval


def test_asr_segment_from_whisper():
    """Test function for the AsrSegment class's from_whisper method."""
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
    assert asr_segment.time_interval == TimeInterval(
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
    """Test function for the AsrWord class's from_whisper method."""
    word = WhisperWord(
        word="hello",
        start=0.0,
        end=0.5,
        probability=0.5,
    )

    asr_word = AsrWord.from_whisper(word)

    assert asr_word.word == "hello"
    assert asr_word.time_interval == TimeInterval(start=word.start, end=word.end)
    assert asr_word.alignment_confidence == word.probability


def test_sum_asr_transcription_info():
    """Test function for the sum method of the AsrTranscriptionInfo class."""
    info_1 = AsrTranscriptionInfo(language="en", language_confidence=0.9)
    info_2 = AsrTranscriptionInfo(language="de", language_confidence=0.8)

    info_sum = sum([info_1, info_2], AsrTranscriptionInfo())

    assert info_sum.language == "en"
    assert info_sum.language_confidence == 0.9

    info_1 = AsrTranscriptionInfo(language="en", language_confidence=0.9)
    info_2 = AsrTranscriptionInfo(language="en", language_confidence=0.5)

    info_sum = sum([info_1, info_2], AsrTranscriptionInfo())

    assert info_sum.language == "en"
    assert info_sum.language_confidence == 0.7


def test_sum_asr_segments():
    """Test function for the sum method of the AsrSegments class."""
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

    asr_segment_1 = AsrSegment.from_whisper(whisper_segment)
    asr_segment_2 = AsrSegment.from_whisper(whisper_segment)

    segment_1 = AsrSegments([asr_segment_1] * 3)
    segment_2 = AsrSegments([asr_segment_2] * 3)

    segments = sum([segment_1, segment_2], AsrSegments())

    assert len(segments) == 6
    assert segments == [asr_segment_1] * 3 + [asr_segment_2] * 3
    assert segments[:3] == segment_1
    assert segments[3:] == segment_2


def test_sum_asr_transcription():
    """Test function for the sum method of the AsrTranscription class."""
    transcription1 = AsrTranscription(text="Hello world")
    transcription2 = AsrTranscription(text="Another transcription")

    transcription = sum([transcription1, transcription2], AsrTranscription())

    assert transcription.text == "Hello world\nAnother transcription"
