from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Annotated, TypedDict

from pydantic import Field

from aana.api.api_generation import Endpoint
from aana.core.models.asr import (
    AsrSegments,
    AsrTranscription,
    AsrTranscriptionInfo,
)
from aana.core.models.chat import Question
from aana.core.models.media import MediaId
from aana.core.models.sampling import SamplingParams
from aana.core.models.vad import VadParams
from aana.core.models.video import VideoInput, VideoMetadata, VideoParams, VideoStatus
from aana.core.models.whisper import BatchedWhisperParams, WhisperParams
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.exceptions.db import MediaIdAlreadyExistsException, UnfinishedVideoException
from aana.integrations.external.decord import generate_frames, get_video_duration
from aana.integrations.external.yt_dlp import download_video
from aana.processors.remote import run_remote
from aana.processors.video import extract_audio, generate_combined_timeline
from aana.projects.chat_with_video.const import asr_model_name, captioning_model_name
from aana.projects.chat_with_video.utils import (
    generate_dialog,
)
from aana.storage.models.video import Status
from aana.storage.services.video import (
    check_media_id_exist,
    delete_media,
    get_video_status,
    load_video_captions,
    load_video_metadata,
    load_video_transcription,
    save_video,
    save_video_captions,
    save_video_transcription,
    update_video_status,
)

if TYPE_CHECKING:
    from aana.core.models.audio import Audio
    from aana.core.models.video import Video


class IndexVideoOutput(TypedDict):
    """The output of the transcribe video endpoint."""

    media_id: MediaId
    metadata: VideoMetadata
    transcription: AsrTranscription
    transcription_info: AsrTranscriptionInfo
    segments: AsrSegments

    captions: Annotated[list[str], Field(..., description="Captions")]
    timestamps: Annotated[
        list[float], Field(..., description="Timestamps for each caption in seconds")
    ]

    transcription_id: Annotated[int, Field(..., description="Transcription Id")]
    caption_ids: Annotated[list[int], Field(..., description="Caption Ids")]


class VideoChatEndpointOutput(TypedDict):
    """Video chat endpoint output."""

    completion: Annotated[str, Field(description="Generated text.")]


class LoadVideoMetadataOutput(TypedDict):
    """The output of the load video metadata endpoint."""

    metadata: VideoMetadata


class VideoStatusOutput(TypedDict):
    """The output of the video status endpoint."""

    status: VideoStatus


class DeleteMediaOutput(TypedDict):
    """The output of the delete media endpoint."""

    media_id: MediaId


class SimpleTranscribeVideoOutput(TypedDict):
    """The output of the simple transcribe video endpoint."""

    transcription: AsrTranscription
    transcription_info: AsrTranscriptionInfo
    segments: AsrSegments


class SimpleTranscribeVideoEndpoint(Endpoint):
    """Simple video transcription (no saving)."""

    async def initialize(self):
        """Initialize the endpoint."""
        self.asr_handle = await AanaDeploymentHandle.create("asr_deployment")

    async def run(
        self, video: VideoInput, whisper_params: WhisperParams
    ) -> AsyncGenerator[SimpleTranscribeVideoOutput, None]:
        """Transcribe video."""
        video_obj: Video = await run_remote(download_video)(video_input=video)
        audio: Audio = extract_audio(video=video_obj)

        transcription_list = []
        segments_list = []
        transcription_info_list = []
        async for whisper_output in self.asr_handle.transcribe_stream(
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


class IndexVideoEndpoint(Endpoint):
    """Transcribe video in chunks endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        self.asr_handle = await AanaDeploymentHandle.create("asr_deployment")
        self.vad_handle = await AanaDeploymentHandle.create("vad_deployment")
        self.captioning_handle = await AanaDeploymentHandle.create(
            "captioning_deployment"
        )
        await super().initialize()

    async def run(
        self,
        video: VideoInput,
        video_params: VideoParams,
        whisper_params: BatchedWhisperParams,
        vad_params: VadParams,
    ) -> AsyncGenerator[IndexVideoOutput, None]:
        """Transcribe video in chunks."""
        media_id = video.media_id
        if check_media_id_exist(media_id):
            raise MediaIdAlreadyExistsException(table_name="media", media_id=video)

        video_obj: Video = await run_remote(download_video)(video_input=video)
        video_duration = await run_remote(get_video_duration)(video=video_obj)
        save_video(video=video_obj, duration=video_duration)
        yield {
            "media_id": media_id,
            "metadata": VideoMetadata(
                title=video_obj.title, description=video_obj.description
            ),
        }

        try:
            update_video_status(media_id=media_id, status=Status.RUNNING)
            audio: Audio = extract_audio(video=video_obj)

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

            async for frames_dict in run_remote(generate_frames)(
                video=video_obj, params=video_params
            ):
                if len(frames_dict["frames"]) == 0:
                    break

                timestamps.extend(frames_dict["timestamps"])
                frame_ids.extend(frames_dict["frame_ids"])

                captioning_output = await self.captioning_handle.generate_batch(
                    images=frames_dict["frames"]
                )
                captions.extend(captioning_output["captions"])

                yield {
                    "captions": captioning_output["captions"],
                    "timestamps": frames_dict["timestamps"],
                }

            save_video_transcription_output = save_video_transcription(
                model_name=asr_model_name,
                media_id=video_obj.media_id,
                transcription=transcription,
                segments=segments,
                transcription_info=transcription_info,
            )

            save_video_captions_output = save_video_captions(
                model_name=captioning_model_name,
                media_id=video_obj.media_id,
                captions=captions,
                timestamps=timestamps,
                frame_ids=frame_ids,
            )

            yield {
                "transcription_id": save_video_transcription_output["transcription_id"],
                "caption_ids": save_video_captions_output["caption_ids"],
            }
        except BaseException:
            update_video_status(media_id=media_id, status=Status.FAILED)
            raise
        else:
            update_video_status(media_id=media_id, status=Status.COMPLETED)


class VideoChatEndpoint(Endpoint):
    """Video chat endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        self.llm_handle = await AanaDeploymentHandle.create("llm_deployment")
        await super().initialize()

    async def run(
        self, media_id: MediaId, question: Question, sampling_params: SamplingParams
    ) -> AsyncGenerator[VideoChatEndpointOutput, None]:
        """Run the video chat endpoint."""
        # check to see if video already processed
        video_status = get_video_status(media_id=media_id)
        if video_status != Status.COMPLETED:
            raise UnfinishedVideoException(
                media_id=media_id,
                status=video_status,
                message=f"The video data is not available, status: {video_status}",
            )

        load_video_transcription_output = load_video_transcription(
            media_id=media_id, model_name=asr_model_name
        )

        loaded_video_captions_output = load_video_captions(
            media_id=media_id, model_name=captioning_model_name
        )

        video_metadata = load_video_metadata(media_id=media_id)

        timeline_output = generate_combined_timeline(
            transcription_segments=load_video_transcription_output["segments"],
            captions=loaded_video_captions_output["captions"],
            caption_timestamps=loaded_video_captions_output["timestamps"],
        )

        dialog = generate_dialog(
            metadata=video_metadata,
            timeline=timeline_output["timeline"],
            question=question,
        )

        async for item in self.llm_handle.chat_stream(
            dialog=dialog, sampling_params=sampling_params
        ):
            yield {"completion": item["text"]}


class LoadVideoMetadataEndpoint(Endpoint):
    """Load video metadata endpoint."""

    async def run(self, media_id: MediaId) -> LoadVideoMetadataOutput:
        """Load video metadata."""
        video_metadata: VideoMetadata = load_video_metadata(media_id=media_id)
        return {
            "metadata": video_metadata,
        }


class GetVideoStatusEndpoint(Endpoint):
    """Get video status endpoint."""

    async def run(self, media_id: MediaId) -> VideoStatusOutput:
        """Load video metadata."""
        video_status: Status = get_video_status(media_id=media_id)
        return {
            "status": video_status,
        }


class DeleteMediaEndpoint(Endpoint):
    """Delete media endpoint."""

    async def run(self, media_id: MediaId) -> DeleteMediaOutput:
        """Delete media."""
        delete_media(media_id=media_id)
        return {"media_id": media_id}
