# ruff: noqa: S101
import json
from importlib import resources
from pathlib import Path
from typing import Literal

import pytest

from aana.core.models.asr import (
    AsrSegment,
    AsrTranscription,
    AsrTranscriptionInfo,
)
from aana.core.models.speaker import SpeakerDiarizationSegment
from aana.processors.speaker import (
    SpeakerDiarizationOutput,
    WhisperOutput,
    asr_postprocessing_for_diarization,
)
from aana.tests.utils import verify_deployment_results


@pytest.mark.parametrize("audio_file", ["sd_sample.wav"])
def test_asr_diarization_post_process(audio_file: Literal["sd_sample.wav"]):
    """Test that the ASR output can be processed to generate diarized transcription."""
    # load precomputed asr and diarization outputs
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

    asr_output = WhisperOutput(
        segments=[AsrSegment.model_validate(segment) for segment in asr_op["segments"]],
        transcription_info=AsrTranscriptionInfo.model_validate(
            asr_op["transcription_info"]
        ),
        transcription=AsrTranscription.model_validate(asr_op["transcription"]),
    )

    with Path.open(diar_path, "r") as json_file:
        diar_op = json.load(json_file)

    diar_output = SpeakerDiarizationOutput(
        segments=[
            SpeakerDiarizationSegment.model_validate(segment)
            for segment in diar_op["segments"]
        ],
    )

    processed_transcription = asr_postprocessing_for_diarization(
        diar_output,
        asr_output,
    )

    verify_deployment_results(expected_results_path, processed_transcription)
