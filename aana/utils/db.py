# ruff: noqa: A002
from pathlib import Path
from urllib.parse import urlparse

from sqlalchemy.orm import Session

from aana.configs.db import media_id_type
from aana.models.core.video import Video
from aana.models.db import (
    CaptionEntity,
    MediaEntity,
    MediaType,
    TranscriptEntity,
    VideoEntity,
)
from aana.models.pydantic.asr_output import (
    AsrSegments,
    AsrTranscriptionInfoList,
    AsrTranscriptionList,
)
from aana.models.pydantic.captions import CaptionsList
from aana.repository.datastore.caption_repo import CaptionRepository
from aana.repository.datastore.engine import engine
from aana.repository.datastore.media_repo import MediaRepository
from aana.repository.datastore.transcript_repo import TranscriptRepository
from aana.repository.datastore.video_repo import VideoRepository


# Just using raw utility functions like this isn't a permanent solution, but
# it's good enough for now to validate what we're working on.
def save_video_batch(
    videos: list[Video],  # , video_params: list[VideoParams]
) -> dict:
    """Saves a batch of videos to datastore."""
    raise NotImplementedError("Needs to be fixed.")
    entities = []
    for video_object, _ in zip(videos, videos, strict=True):
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
        entity = MediaEntity(
            id=video_object.media_id,
            media_type=MediaType.VIDEO,
            orig_filename=orig_filename,
            orig_url=orig_url,
        )
        entities.append(entity)
    with Session(engine) as session:
        repo = MediaRepository(session)
        results = repo.create_multiple(entities)
        return {
            "media_ids": [result.id for result in results]  # type: ignore
        }


def save_video_single(
    video: Video,
) -> dict:
    """Saves a batch of videos to datastore."""
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
        media=media_entity,
        media_type=MediaType.VIDEO,
        orig_filename=orig_filename,
        orig_url=orig_url,
    )
    with Session(engine) as session:
        media_repo = MediaRepository(session)
        _ = media_repo.create(media_entity)
        video_repo = VideoRepository(session)
        _ = video_repo.create(video_entity)
        return {
            "media_id": media_entity.id,  # type: ignore
            "video_id": video_entity.id,
        }


def save_captions_batch(
    media_ids: list[media_id_type],
    model_name: str,
    captions_list: list[CaptionsList],
    timestamps_list: list[list[float]],
    frame_ids_list: list[list[int]],
) -> dict:
    """Save captions."""
    raise NotImplementedError("Needs to be fixed")

    with Session(engine) as session:
        entities = [
            CaptionEntity.from_caption_output(
                model_name, media_id, frame_id, timestamp, caption
            )
            for media_id, captions, timestamps, frame_ids in zip(
                media_ids, captions_list, timestamps_list, frame_ids_list, strict=True
            )
            for caption, timestamp, frame_id in zip(
                captions, timestamps[:-1], frame_ids, strict=True
            )
        ]
        repo = CaptionRepository(session)
        results = repo.create_multiple(entities)
        # return [c.id for c in results]
        return {
            "caption_ids": [c.id for c in results]  # type: ignore
        }


def save_video_captions(
    model_name: str,
    media_id: media_id_type,
    video_id: int,
    captions: CaptionsList,
    timestamps: list[float],
    frame_ids: list[int],
) -> dict:
    """Save captions."""
    with Session(engine) as session:
        entities = [
            CaptionEntity.from_caption_output(
                model_name, media_id, video_id, frame_id, timestamp, caption
            )
            for caption, timestamp, frame_id in zip(
                captions, timestamps, frame_ids, strict=True
            )
        ]
        repo = CaptionRepository(session)
        results = repo.create_multiple(entities)
        # return [c.id for c in results]
        return {
            "caption_ids": [c.id for c in results]  # type: ignore
        }


def save_transcripts_batch(
    model_name: str,
    media_ids: list[media_id_type],
    transcript_info_list: list[AsrTranscriptionInfoList],
    transcripts_list: list[AsrTranscriptionList],
    segments_list: list[AsrSegments],
) -> dict:
    """Save transcripts."""
    raise NotImplementedError("Needs to be fixed")
    with Session(engine) as session:
        entities = [
            TranscriptEntity.from_asr_output(model_name, media_id, info, txn, seg)
            for media_id, transcript_infos, transcripts, segments in zip(
                media_ids,
                transcript_info_list,
                transcripts_list,
                segments_list,
                strict=True,
            )
            for info, txn, seg in zip(
                transcript_infos, transcripts, segments, strict=True
            )
        ]

        repo = TranscriptRepository(session)
        entities = repo.create_multiple(entities)
        return {
            "transcript_ids": [c.id for c in entities]  # type: ignore
        }


def save_video_transcripts(
    model_name: str,
    media_id: media_id_type,
    video_id: int,
    transcript_infos: AsrTranscriptionInfoList,
    transcripts: AsrTranscriptionList,
    segments: AsrSegments,
) -> dict:
    """Save transcripts."""
    with Session(engine) as session:
        entities = [
            TranscriptEntity.from_asr_output(
                model_name, media_id, video_id, info, txn, seg
            )
            for info, txn, seg in zip(
                transcript_infos, transcripts, segments, strict=True
            )
        ]

        repo = TranscriptRepository(session)
        entities = repo.create_multiple(entities)
        print(len(entities))
        return {
            "transcript_ids": [c.id for c in entities]  # type: ignore
        }
