# ruff: noqa: A002
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlparse

from sqlalchemy.orm import Session

from aana.core.models.asr import (
    AsrSegment,
    AsrSegments,
    AsrTranscription,
    AsrTranscriptionInfo,
)
from aana.core.models.captions import CaptionsList
from aana.core.models.media import MediaId
from aana.core.models.video import Video, VideoMetadata
from aana.storage.engine import engine
from aana.storage.models import (
    CaptionEntity,
    MediaEntity,
    MediaType,
    TranscriptEntity,
    VideoEntity,
)
from aana.storage.models.video import Status as VideoStatus
from aana.storage.repository.caption import CaptionRepository
from aana.storage.repository.media import MediaRepository
from aana.storage.repository.transcript import TranscriptRepository
from aana.storage.repository.video import VideoRepository


# Just using raw utility functions like this isn't a permanent solution, but
# it's good enough for now to validate what we're working on.
def save_video_batch(
    videos: list[Video],
    durations: list[float],
    # , video_params: list[VideoParams]
) -> dict:
    """Saves a batch of videos to datastore.

    Args:
        videos (list[Video]): The video objects.
        durations (list[float]): the duration of each video object

    Returns:
        dict: The dictionary with video and media IDs.
    """
    media_entities = []
    video_entities = []
    for video_object, duration in zip(videos, durations, strict=True):
        if video_object.url is not None:
            orig_url = video_object.url
            parsed_url = urlparse(orig_url)
            orig_filename = Path(parsed_url.path).name
        elif video_object.path is not None:
            parsed_path = Path(video_object.path)
            orig_filename = parsed_path.name
            orig_url = None
        else:
            orig_url = None
            orig_filename = None
        media_entity = MediaEntity(
            id=video_object.media_id,
            media_type=MediaType.VIDEO,
        )
        video_entity = VideoEntity(
            media=media_entity,
            duration=duration,
            orig_filename=orig_filename,
            orig_url=orig_url,
        )
        media_entities.append(media_entity)
        video_entities.append(video_entity)
    with Session(engine) as session:
        m_repo = MediaRepository(session)
        v_repo = VideoRepository(session)
        media_entities = m_repo.create_multiple(media_entities)
        video_entities = v_repo.create_multiple(video_entities)
        return {"media_ids": [m.id for m in media_entities]}


def save_video(video: Video, duration: float) -> dict:
    """Saves a video to datastore.

    Args:
        video (Video): The video object.
        duration (float): the duration of the video object

    Returns:
        dict: The dictionary with video and media IDs.
    """
    if video.url is not None:
        orig_url = video.url
        parsed_url = urlparse(orig_url)
        orig_filename = Path(parsed_url.path).name
    elif video.path is not None:
        parsed_path = Path(video.path)
        orig_filename = parsed_path.name
        orig_url = None
    else:
        orig_url = None
        orig_filename = None
    media_entity = MediaEntity(id=video.media_id, media_type=MediaType.VIDEO)
    video_entity = VideoEntity(
        id=video.media_id,
        orig_filename=orig_filename,
        orig_url=orig_url,
        title=video.title,
        duration=duration,
        description=video.description,
    )
    with Session(engine) as session:
        media_repo = MediaRepository(session)
        _ = media_repo.create(media_entity)
        video_repo = VideoRepository(session)
        _ = video_repo.create(video_entity)
        return {
            "media_id": media_entity.id,
        }


def get_video_status(
    media_id: MediaId,
) -> VideoStatus:
    """Load video status from database.

    Args:
        media_id (MediaId): The media ID.

    Returns:
        VideoStatus: The video status.
    """
    with Session(engine) as session:
        entity: VideoEntity = VideoRepository(session).read(media_id)
        return entity.status


def check_media_id_exist(
    media_id: MediaId,
) -> bool:
    """Check if media_id exists in the database.

    Args:
        media_id (MediaId): The media ID.

    Returns:
        bool: True if the media_id exists, False otherwise.
    """
    with Session(engine) as session:
        return MediaRepository(session).check_media_exists(media_id)


def update_video_status(media_id: MediaId, status: VideoStatus):
    """Update the video status.

    Args:
        media_id (str): The media ID.
        status (VideoStatus): The new status.
    """
    with Session(engine) as session:
        video_repo = VideoRepository(session)
        media_entity = video_repo.read(media_id)
        media_entity.status = status
        session.commit()


def delete_media(media_id: MediaId) -> dict:
    """Deletes a media file from the database.

    Args:
        media_id (MediaId): The media ID.

    Returns:
        dict: The dictionary with the ID of the deleted entity.
    """
    with Session(engine) as session:
        media_repo = MediaRepository(session)
        media_entity = media_repo.delete(media_id, check=True)
    return {"media_id": media_entity.id}


class SaveVideoCaptionOutput(TypedDict):
    """The output of the save video caption endpoint."""

    caption_ids: list[int]


def save_video_captions(
    model_name: str,
    media_id: MediaId,
    captions: CaptionsList,
    timestamps: list[float],
    frame_ids: list[int],
) -> SaveVideoCaptionOutput:
    """Save captions.

    Args:
        model_name (str): The name of the model used to generate the captions.
        media_id (MediaId): the media ID of the video.
        captions (CaptionsList): The captions.
        timestamps (list[float]): The timestamps.
        frame_ids (list[int]): The frame IDs.

    Returns:
        SaveVideoCaptionOutput: The dictionary with the IDs of the saved entities.
    """
    with Session(engine) as session:
        entities = [
            CaptionEntity.from_caption_output(
                model_name, media_id, frame_id, timestamp, caption
            )
            for caption, timestamp, frame_id in zip(
                captions, timestamps, frame_ids, strict=True
            )
        ]
        repo = CaptionRepository(session)
        results = repo.create_multiple(entities)
        return {
            "caption_ids": [c.id for c in results]  # type: ignore
        }


def save_video_transcription(
    model_name: str,
    media_id: MediaId,
    transcription_info: AsrTranscriptionInfo,
    transcription: AsrTranscription,
    segments: AsrSegments,
) -> dict:
    """Save transcripts.

    Args:
        model_name (str): The name of the model used to generate the transcript.
        media_id (MediaId): The media id of the video
        transcription_info (AsrTranscriptionInfo): The ASR transcription info.
        transcription (AsrTranscription): The ASR transcription.
        segments (AsrSegments): The ASR segments.

    Returns:
        dict: The dictionary with the IDs of the saved entities.
    """
    with Session(engine) as session:
        entities = [
            TranscriptEntity.from_asr_output(
                model_name=model_name,
                media_id=media_id,
                transcription=transcription,
                segments=segments,
                info=transcription_info,
            )
        ]

        repo = TranscriptRepository(session)
        entities = repo.create_multiple(entities)
        transcription_id = entities[0].id

        return {
            "transcription_id": transcription_id,
        }


def save_transcripts_batch(
    model_name: str,
    media_ids: list[MediaId],
    transcription_info_list: list[AsrTranscriptionInfo],
    transcription_list: list[AsrTranscription],
    segments_list: list[list[AsrSegment]],
) -> dict:
    """Save transcripts in a batch.

    Arguments:
        model_name (str): the nameof the model used to generate the batch of transcripts
        media_ids (list[MediaId]): the media ids of each video in the batch
        transcription_info_list (list[AsrTranscriptionInfo]): list of transcript metadata
        transcription_list (list[AsrTranscription]): list of transcripts
        segments_list (list[list[AsrSegment]]): list of audio segment definitions

    Returns:
        dict with key "transcription_ids"
    """
    with Session(engine) as session:
        entities = [
            TranscriptEntity.from_asr_output(
                model_name,
                media_id,
                transcript_info,
                transcript,
                segments,
            )
            for media_id, transcript_info, transcript, segments in zip(
                media_ids,
                transcription_info_list,
                transcription_list,
                segments_list,
                strict=True,
            )
        ]
        repo = TranscriptRepository(session)
        entities = repo.create_multiple(entities)
        return {
            "transcription_ids": [c.id for c in entities]  # type: ignore
        }


class LoadTranscriptionOutput(TypedDict):
    """The output of the load transcription endpoint."""

    transcription: AsrTranscription
    segments: AsrSegments
    transcription_info: AsrTranscriptionInfo


def load_video_transcription(
    model_name: str,
    media_id: MediaId,
) -> LoadTranscriptionOutput:
    """Load transcript from database.

    Args:
        model_name (str): The name of the model used to generate the transcript.
        media_id (MediaId): The media ID.

    Returns:
        LoadTranscriptionOutput: The dictionary with the transcript, segments, and info.
    """
    with Session(engine) as session:
        repo = TranscriptRepository(session)
        entity: TranscriptEntity = repo.get_transcript(model_name, media_id)
        transcription = AsrTranscription(text=entity.transcript)
        segments = [AsrSegment(**s) for s in entity.segments]
        info = AsrTranscriptionInfo(
            language=entity.language,
            language_confidence=entity.language_confidence,
        )
        return {
            "transcription": transcription,
            "segments": segments,
            "transcription_info": info,
        }


class LoadVideoCaptionsOutput(TypedDict):
    """The output of the load video captions endpoint."""

    captions: CaptionsList
    timestamps: list[float]
    frame_ids: list[int]


def load_video_captions(
    model_name: str,
    media_id: MediaId,
) -> LoadVideoCaptionsOutput:
    """Load captions from database.

    Args:
        model_name (str): The name of the model used to generate the captions.
        media_id (MediaId): The media ID.

    Returns:
        LoadVideoCaptionsOutput: The dictionary with the captions, timestamps, and frame IDs.
            captions: CaptionsList
            timestamps: list[float]
            frame_ids: list[int]
    """
    with Session(engine) as session:
        repo = CaptionRepository(session)
        entities = repo.get_captions(model_name, media_id)
        captions = [c.caption for c in entities]
        timestamps = [c.timestamp for c in entities]
        frame_ids = [c.frame_id for c in entities]
        return {
            "captions": captions,
            "timestamps": timestamps,
            "frame_ids": frame_ids,
        }


def load_video_metadata(
    media_id: MediaId,
) -> VideoMetadata:
    """Load video metadata from database.

    Args:
        media_id (MediaId): The media ID.

    Returns:
        VideoMetadata: The video metadata.
    """
    with Session(engine) as session:
        video_entity = VideoRepository(session).read(media_id)
        return VideoMetadata(
            title=video_entity.title,
            description=video_entity.description,
        )
