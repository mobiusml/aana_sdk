# ruff: noqa: S101
import json
from importlib import resources
from pathlib import Path
from typing import Literal

import pytest

from aana.core.models.asr import AsrSegment
from aana.core.models.speaker import SpeakerDiarizationSegment
from aana.processors.speaker import ASRPostProcessingForDiarization
from aana.tests.utils import verify_deployment_results


@pytest.mark.parametrize("audio_file", ["sd_sample.wav"])
def test_asr_diarization_post_process(audio_file: Literal["sd_sample.wav"]):
    """Test that the ASR output can be processed to generate diarized transcription."""
    # load precomputed ASR and Diarization outputs
    asr_path = (
        resources.files("aana.tests.files.expected.whisper")
        / f"whisper_medium_{audio_file}.json"
    )
    diar_path = (
        resources.files("aana.tests.files.expected.sd")
        / f"{Path(audio_file).stem}.json"
    )
    expected_results_path = (
        resources.files("aana.tests.files.expected.whisper")
        / f"whisper_medium_{audio_file}_diar.json"
    )

    # convert to WhisperOutput and SpeakerDiarizationOutput
    with Path.open(asr_path, "r") as json_file:
        asr_op = json.load(json_file)

    asr_segments = [
        AsrSegment.model_validate(segment) for segment in asr_op["segments"]
    ]

    with Path.open(diar_path, "r") as json_file:
        diar_op = json.load(json_file)

    diarized_segments = [
        SpeakerDiarizationSegment.model_validate(segment)
        for segment in diar_op["segments"]
    ]
    post_processor = ASRPostProcessingForDiarization(
        diarized_segments=diarized_segments, transcription_segments=asr_segments
    )
    asr_op["segments"] = post_processor.process()

    verify_deployment_results(expected_results_path, asr_op)
