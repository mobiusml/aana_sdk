import json
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
from aana.exceptions.io import VideoTooLongException
from aana.integrations.external.decord import generate_frames, get_video_duration
from aana.integrations.external.yt_dlp import download_video, get_video_metadata
from aana.processors.remote import run_remote
from aana.processors.video import extract_audio, generate_combined_timeline
from aana.projects.chat_with_video.const import (
    asr_model_name,
    captioning_model_name,
    max_video_len,
)
from aana.projects.chat_with_video.utils import (
    generate_dialog,
)
from aana.storage.models.extended_video import VideoProcessingStatus
from aana.storage.models.video import Status
from aana.storage.repository.extended_video import ExtendedVideoRepository
from aana.storage.repository.extended_video_caption import (
    ExtendedVideoCaptionRepository,
)
from aana.storage.repository.extended_video_transcript import (
    ExtendedVideoTranscriptRepository,
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
        await super().initialize()
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
        await super().initialize()
        self.asr_handle = await AanaDeploymentHandle.create("asr_deployment")
        self.vad_handle = await AanaDeploymentHandle.create("vad_deployment")
        self.captioning_handle = await AanaDeploymentHandle.create(
            "captioning_deployment"
        )
        self.extended_video_repo = ExtendedVideoRepository(self.session)
        self.transcript_repo = ExtendedVideoTranscriptRepository(self.session)
        self.caption_repo = ExtendedVideoCaptionRepository(self.session)

    async def run(  # noqa: C901
        self,
        video: VideoInput,
        video_params: VideoParams,
        whisper_params: BatchedWhisperParams,
        vad_params: VadParams,
    ) -> AsyncGenerator[IndexVideoOutput, None]:
        """Transcribe video in chunks."""
        media_id = video.media_id
        if self.extended_video_repo.check_media_exists(media_id):
            raise MediaIdAlreadyExistsException(table_name="media", media_id=video)

        video_duration = None
        if video.url is not None:
            video_metadata = get_video_metadata(video.url)
            video_duration = video_metadata.duration

        # precheck for max video length before actually download the video if possible
        if video_duration and video_duration > max_video_len:
            raise VideoTooLongException(
                video=video,
                video_len=video_duration,
                max_len=max_video_len,
            )

        video_obj: Video = await run_remote(download_video)(video_input=video)
        if video_duration is None:
            video_duration = await run_remote(get_video_duration)(video=video_obj)

        if video_duration > max_video_len:
            raise VideoTooLongException(
                video=video_obj,
                video_len=video_duration,
                max_len=max_video_len,
            )

        self.extended_video_repo.save(video=video_obj, duration=video_duration)
        yield {
            "media_id": media_id,
            "metadata": VideoMetadata(
                title=video_obj.title,
                description=video_obj.description,
                duration=video_duration,
            ),
        }

        try:
            self.extended_video_repo.update_status(
                media_id, VideoProcessingStatus.RUNNING
            )
            audio: Audio = extract_audio(video=video_obj)

            # TODO: Update once batched whisper PR is merged
            # vad_output = await self.vad_handle.asr_preprocess_vad(
            #     audio=audio, params=vad_params
            # )
            # vad_segments = vad_output["segments"]

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

            transcription_entity = self.transcript_repo.save(
                model_name=asr_model_name,
                media_id=video_obj.media_id,
                transcription=transcription,
                segments=segments,
                transcription_info=transcription_info,
            )

            caption_entities = self.caption_repo.save_all(
                model_name=captioning_model_name,
                media_id=video_obj.media_id,
                captions=captions,
                timestamps=timestamps,
                frame_ids=frame_ids,
            )

            yield {
                "transcription_id": transcription_entity.id,
                "caption_ids": [c.id for c in caption_entities],
            }
        except BaseException:
            self.extended_video_repo.update_status(
                media_id, VideoProcessingStatus.FAILED
            )
            raise
        else:
            self.extended_video_repo.update_status(
                media_id, VideoProcessingStatus.COMPLETED
            )


class VideoChatEndpoint(Endpoint):
    """Video chat endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        await super().initialize()
        self.llm_handle = await AanaDeploymentHandle.create("llm_deployment")
        self.transcript_repo = ExtendedVideoTranscriptRepository(self.session)
        self.caption_repo = ExtendedVideoCaptionRepository(self.session)
        self.video_repo = ExtendedVideoRepository(self.session)

    async def run(
        self, media_id: MediaId, question: Question, sampling_params: SamplingParams
    ) -> AsyncGenerator[VideoChatEndpointOutput, None]:
        """Run the video chat endpoint."""
        # check to see if video already processed
        video_status = self.video_repo.get_status(media_id)
        if video_status != Status.COMPLETED:
            raise UnfinishedVideoException(
                media_id=media_id,
                status=video_status,
                message=f"The video data is not available, status: {video_status}",
            )

        transcription_output = self.transcript_repo.get_transcript(
            model_name=asr_model_name, media_id=media_id
        )

        captions_output = self.caption_repo.get_captions(
            model_name=captioning_model_name, media_id=media_id
        )

        video_metadata = self.video_repo.get_metadata(media_id)

        timeline_output = generate_combined_timeline(
            transcription_segments=transcription_output["segments"],
            captions=captions_output["captions"],
            caption_timestamps=captions_output["timestamps"],
        )
        timeline_json = json.dumps(
            timeline_output["timeline"], indent=4, separators=(",", ": ")
        )

        dialog = generate_dialog(
            metadata=video_metadata,
            timeline=timeline_json,
            question=question,
        )
        async for item in self.llm_handle.chat_stream(
            dialog=dialog, sampling_params=sampling_params
        ):
            yield {"completion": item["text"]}


class LoadVideoMetadataEndpoint(Endpoint):
    """Load video metadata endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        await super().initialize()
        self.video_repo = ExtendedVideoRepository(self.session)

    async def run(self, media_id: MediaId) -> LoadVideoMetadataOutput:
        """Load video metadata."""
        video_metadata = self.video_repo.get_metadata(media_id)
        return {
            "metadata": video_metadata,
        }


class GetVideoStatusEndpoint(Endpoint):
    """Get video status endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        await super().initialize()
        self.video_repo = ExtendedVideoRepository(self.session)

    async def run(self, media_id: MediaId) -> VideoStatusOutput:
        """Load video metadata."""
        video_status = self.video_repo.get_status(media_id)
        return {
            "status": video_status,
        }


class DeleteMediaEndpoint(Endpoint):
    """Delete media endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        await super().initialize()
        self.video_repo = ExtendedVideoRepository(self.session)

    async def run(self, media_id: MediaId) -> DeleteMediaOutput:
        """Delete media."""
        self.video_repo.delete(media_id)
        return {"media_id": media_id}
