from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Annotated, TypedDict

from pydantic import Field

from aana.api.api_generation import Endpoint
from aana.core.models.asr import (
    AsrSegments,
    AsrTranscription,
    AsrTranscriptionInfo,
)
from aana.core.models.media import MediaId
from aana.core.models.video import VideoInput
from aana.core.models.whisper import WhisperParams
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.integrations.external.yt_dlp import download_video
from aana.processors.remote import run_remote
from aana.processors.video import extract_audio
from aana.projects.whisper.const import asr_model_name
from aana.storage.repository.extended_video_transcript import (
    ExtendedVideoTranscriptRepository,
)
from aana.storage.repository.media import MediaRepository
from aana.storage.repository.video import VideoRepository

if TYPE_CHECKING:
    from aana.core.models.audio import Audio
    from aana.core.models.video import Video


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


class TranscribeVideoEndpoint(Endpoint):
    """Transcribe video endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        await super().initialize()
        self.asr_handle = await AanaDeploymentHandle.create("asr_deployment")
        self.transcript_repo = ExtendedVideoTranscriptRepository(self.session)
        self.video_repo = VideoRepository(self.session)

    async def run(
        self, video: VideoInput, whisper_params: WhisperParams
    ) -> AsyncGenerator[TranscribeVideoOutput, None]:
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
        transcription = sum(transcription_list, AsrTranscription())
        segments = sum(segments_list, AsrSegments())
        transcription_info = sum(transcription_info_list, AsrTranscriptionInfo())

        self.video_repo.save(video_obj)

        transcription_entity = self.transcript_repo.save(
            model_name=asr_model_name,
            media_id=video_obj.media_id,
            transcription=transcription,
            segments=segments,
            transcription_info=transcription_info,
        )
        yield {"transcription_id": transcription_entity.id}


# TODO: Update once batched whisper PR is merged
# class TranscribeVideoInChunksEndpoint(Endpoint):
#     """Transcribe video in chunks endpoint."""

#     async def initialize(self):
#         """Initialize the endpoint."""
#         self.asr_handle = await AanaDeploymentHandle.create("asr_deployment")
#         self.vad_handle = await AanaDeploymentHandle.create("vad_deployment")
#         await super().initialize()

#     async def run(
#         self,
#         video: VideoInput,
#         whisper_params: BatchedWhisperParams,
#         vad_params: VadParams,
#     ) -> AsyncGenerator[TranscribeVideoOutput, None]:
#         """Transcribe video in chunks."""
#         video_obj: Video = await run_remote(download_video)(video_input=video)
#         audio: Audio = extract_audio(video=video_obj)

#         vad_output = await self.vad_handle.asr_preprocess_vad(
#             audio=audio, params=vad_params
#         )
#         vad_segments = vad_output["segments"]

#         transcription_list = []
#         segments_list = []
#         transcription_info_list = []
#         async for whisper_output in self.asr_handle.transcribe_in_chunks(
#             audio=audio, params=whisper_params, segments=vad_segments
#         ):
#             transcription_list.append(whisper_output["transcription"])
#             segments_list.append(whisper_output["segments"])
#             transcription_info_list.append(whisper_output["transcription_info"])
#             yield {
#                 "transcription": whisper_output["transcription"],
#                 "segments": whisper_output["segments"],
#                 "info": whisper_output["transcription_info"],
#             }
#         transcription = sum(transcription_list, AsrTranscription())
#         segments = sum(segments_list, AsrSegments())
#         transcription_info = sum(transcription_info_list, AsrTranscriptionInfo())

#         save_video(
#             video=video_obj, duration=0.0
#         )  # set duration to 0.0 as we don't have the actual duration

#         transcription_record = save_video_transcription(
#             model_name=asr_model_name,
#             media_id=video_obj.media_id,
#             transcription=transcription,
#             segments=segments,
#             transcription_info=transcription_info,
#         )
#         yield {"transcription_id": transcription_record["transcription_id"]}


class LoadTranscriptionEndpoint(Endpoint):
    """Load transcription endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        await super().initialize()
        self.transcript_repo = ExtendedVideoTranscriptRepository(self.session)

    async def run(self, media_id: MediaId) -> TranscriptionOutput:
        """Load transcription."""
        transcription_record = self.transcript_repo.get_transcript(
            model_name=asr_model_name, media_id=media_id
        )
        return {
            "transcription": transcription_record["transcription"],
            "segments": transcription_record["segments"],
            "info": transcription_record["transcription_info"],
        }


class DeleteMediaEndpoint(Endpoint):
    """Delete media endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        await super().initialize()
        self.media_repo = MediaRepository(self.session)

    async def run(self, media_id: MediaId) -> DeleteMediaOutput:
        """Delete media."""
        self.media_repo.delete(media_id)
        return {"media_id": media_id}
