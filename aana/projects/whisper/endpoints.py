from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Annotated, TypedDict

from pydantic import Field

from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.models.pydantic.asr_output import (
    AsrSegments,
    AsrTranscription,
    AsrTranscriptionInfo,
)
from aana.models.pydantic.media_id import MediaId
from aana.models.pydantic.vad_params import VadParams
from aana.models.pydantic.video_input import VideoInput
from aana.models.pydantic.whisper_params import WhisperParams
from aana.projects.whisper.const import asr_model_name
from aana.utils.db import (
    delete_media,
    load_video_transcription,
    save_video,
    save_video_transcription,
)
from aana.utils.video import download_video, extract_audio

if TYPE_CHECKING:
    from aana.models.core.audio import Audio
    from aana.models.core.video import Video


class TranscribeVideoOutput(TypedDict):
    """The output of the transcribe video endpoint."""

    transcription: AsrTranscription
    transcription_info: AsrTranscriptionInfo
    segments: AsrSegments
    transcription_id: Annotated[int, Field(..., description="Transcription Id")]


class TranscriptionOutput(TypedDict):
    """The transcription output."""

    transcription: AsrTranscription
    segments: AsrSegments
    transcription_info: AsrTranscriptionInfo


class DeleteMediaOutput(TypedDict):
    """The output of the delete media endpoint."""

    media_id: MediaId


async def transcribe_video_endpoint(
    video_input: VideoInput, whisper_params: WhisperParams
) -> AsyncGenerator[TranscribeVideoOutput, None]:
    """Transcribe video."""
    asr_handle = await AanaDeploymentHandle.create("asr_deployment")

    video: Video = download_video(video_input=video_input)
    audio: Audio = extract_audio(video=video)

    transcription_list = []
    segments_list = []
    transcription_info_list = []
    async for whisper_output in asr_handle.transcribe_stream(
        audio=audio, params=whisper_params
    ):
        transcription_list.append(whisper_output["transcription"])
        segments_list.append(whisper_output["segments"])
        transcription_info_list.append(whisper_output["transcription_info"])
        yield {
            "transcription": whisper_output["transcription"],
            "segments": whisper_output["segments"],
            "info": whisper_output["transcription_info"],
        }
    transcription = sum(transcription_list, AsrTranscription())
    segments = sum(segments_list, AsrSegments())
    transcription_info = sum(transcription_info_list, AsrTranscriptionInfo())

    save_video(
        video=video, duration=0.0
    )  # set duration to 0.0 as we don't have the actual duration

    transcription_record = save_video_transcription(
        model_name=asr_model_name,
        media_id=video.media_id,
        transcription=transcription,
        segments=segments,
        transcription_info=transcription_info,
    )
    yield {"transcription_id": transcription_record["transcription_id"]}


async def transcribe_video_in_chunks_endpoint(
    video_input: VideoInput, whisper_params: WhisperParams, vad_params: VadParams
) -> AsyncGenerator[TranscribeVideoOutput, None]:
    """Transcribe video in chunks."""
    asr_handle = await AanaDeploymentHandle.create("asr_deployment")
    vad_handle = await AanaDeploymentHandle.create("vad_deployment")

    video: Video = download_video(video_input=video_input)
    audio: Audio = extract_audio(video=video)

    vad_output = await vad_handle.asr_preprocess_vad(audio=audio, params=vad_params)
    vad_segments = vad_output["segments"]

    transcription_list = []
    segments_list = []
    transcription_info_list = []
    async for whisper_output in asr_handle.transcribe_in_chunks(
        audio=audio, params=whisper_params, segments=vad_segments
    ):
        transcription_list.append(whisper_output["transcription"])
        segments_list.append(whisper_output["segments"])
        transcription_info_list.append(whisper_output["transcription_info"])
        yield {
            "transcription": whisper_output["transcription"],
            "segments": whisper_output["segments"],
            "info": whisper_output["transcription_info"],
        }
    transcription = sum(transcription_list, AsrTranscription())
    segments = sum(segments_list, AsrSegments())
    transcription_info = sum(transcription_info_list, AsrTranscriptionInfo())

    save_video(
        video=video, duration=0.0
    )  # set duration to 0.0 as we don't have the actual duration

    transcription_record = save_video_transcription(
        model_name=asr_model_name,
        media_id=video.media_id,
        transcription=transcription,
        segments=segments,
        transcription_info=transcription_info,
    )
    yield {"transcription_id": transcription_record["transcription_id"]}


async def load_transcription_endpoint(media_id: MediaId) -> TranscriptionOutput:
    """Load transcription."""
    transcription_record = load_video_transcription(
        model_name=asr_model_name, media_id=media_id
    )
    return {
        "transcription": transcription_record["transcription"],
        "segments": transcription_record["segments"],
        "transcription_info": transcription_record["transcription_info"],
    }


async def delete_media_endpoint(media_id: MediaId) -> DeleteMediaOutput:
    """Delete media."""
    delete_media(media_id=media_id)
    return {"media_id": media_id}
