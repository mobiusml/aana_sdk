from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Annotated, TypedDict

from pydantic import Field

from aana.api.api_generation import Endpoint
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.models.pydantic.asr_output import (
    AsrSegments,
    AsrTranscription,
    AsrTranscriptionInfo,
)
from aana.models.pydantic.vad_params import VadParams
from aana.models.pydantic.video_input import VideoInput
from aana.models.pydantic.video_params import VideoParams
from aana.models.pydantic.whisper_params import WhisperParams
from aana.projects.whisper.const import asr_model_name, captioning_model_name
from aana.utils.db import save_video, save_video_captions, save_video_transcription
from aana.utils.general import run_remote
from aana.utils.video import download_video, extract_audio, generate_frames_decord

if TYPE_CHECKING:
    from aana.models.core.audio import Audio
    from aana.models.core.video import Video


class IndexVideoOutput(TypedDict):
    """The output of the transcribe video endpoint."""

    transcription: AsrTranscription
    transcription_info: AsrTranscriptionInfo
    segments: AsrSegments

    captions: Annotated[list[str], Field(..., description="Captions")]
    timestamps: Annotated[
        list[float], Field(..., description="Timestamps for each caption in seconds")
    ]

    transcription_id: Annotated[int, Field(..., description="Transcription Id")]
    caption_ids: Annotated[list[int], Field(..., description="Caption Ids")]


class IndexVideoEndpoint(Endpoint):
    """Transcribe video in chunks endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        self.asr_handle = await AanaDeploymentHandle.create("asr_deployment")
        self.vad_handle = await AanaDeploymentHandle.create("vad_deployment")
        self.captioning_handle = await AanaDeploymentHandle.create(
            "captioning_deployment"
        )

    async def run(
        self,
        video_input: VideoInput,
        video_params: VideoParams,
        whisper_params: WhisperParams,
        vad_params: VadParams,
    ) -> AsyncGenerator[IndexVideoOutput, None]:
        """Transcribe video in chunks."""
        video: Video = await run_remote(download_video)(video_input=video_input)
        audio: Audio = extract_audio(video=video)

        vad_output = await self.vad_handle.asr_preprocess_vad(
            audio=audio, params=vad_params
        )
        vad_segments = vad_output["segments"]

        transcription_list = []
        segments_list = []
        transcription_info_list = []
        async for whisper_output in self.asr_handle.transcribe_in_chunks(
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

        captions = []
        timestamps = []
        frame_ids = []
        video_duration = 0.0
        async for frames_dict in run_remote(generate_frames_decord)(
            video=video, params=video_params
        ):
            timestamps.extend(frames_dict["timestamps"])
            frame_ids.extend(frames_dict["frame_ids"])
            video_duration = frames_dict["duration"]

            captioning_output = await self.captioning_handle.generate_batch(
                images=frames_dict["frames"]
            )
            captions.extend(captioning_output["captions"])

            yield {
                "captions": captioning_output["captions"],
                "timestamps": frames_dict["timestamps"],
            }

        save_video(video=video, duration=video_duration)

        save_video_transcription_output = save_video_transcription(
            model_name=asr_model_name,
            media_id=video.media_id,
            transcription=transcription,
            segments=segments,
            transcription_info=transcription_info,
        )

        save_video_captions_output = save_video_captions(
            model_name=captioning_model_name,
            media_id=video.media_id,
            captions=captions,
            timestamps=timestamps,
            frame_ids=frame_ids,
        )

        yield {
            "transcription_id": save_video_transcription_output["transcription_id"],
            "caption_ids": save_video_captions_output["caption_ids"],
        }
