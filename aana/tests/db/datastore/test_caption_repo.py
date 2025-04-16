# ruff: noqa: S101

import random
import uuid

import pytest

from aana.core.models.captions import Caption
from aana.exceptions.db import NotFoundException
from aana.storage.repository.caption import CaptionRepository


@pytest.fixture(scope="function")
def dummy_caption():
    """Creates a dummy caption for testing."""
    caption = Caption(f"This is a caption {uuid.uuid4()}")
    frame_id = random.randint(0, 100)  # noqa: S311
    timestamp = random.random()  # noqa: S311
    return caption, frame_id, timestamp


@pytest.mark.asyncio
async def test_save_caption(db_session_manager, dummy_caption):
    """Tests saving a caption."""
    async with db_session_manager.session() as session:
        caption, frame_id, timestamp = dummy_caption
        model_name = "blip2"

        caption_repo = CaptionRepository(session)
        caption_entity = await caption_repo.save(
            model_name=model_name,
            caption=caption,
            frame_id=frame_id,
            timestamp=timestamp,
        )
        caption_id = caption_entity.id

        caption_entity = await caption_repo.read(caption_id)
        assert caption_entity.model == model_name
        assert caption_entity.frame_id == frame_id
        assert caption_entity.timestamp == timestamp
        assert caption_entity.caption == caption

        await caption_repo.delete(caption_id)
        with pytest.raises(NotFoundException):
            await caption_repo.read(caption_id)


@pytest.mark.asyncio
async def test_save_all_captions(db_session_manager, dummy_caption):
    """Tests saving all captions."""
    async with db_session_manager.session() as session:
        captions, frame_ids, timestamps = [], [], []
        for _ in range(3):
            caption, frame_id, timestamp = dummy_caption
            captions.append(caption)
            frame_ids.append(frame_id)
            timestamps.append(timestamp)
        model_name = "blip2"

        caption_repo = CaptionRepository(session)
        caption_entities = await caption_repo.save_all(
            model_name=model_name,
            captions=captions,
            timestamps=timestamps,
            frame_ids=frame_ids,
        )
        assert len(caption_entities) == len(captions)

        caption_ids = [caption_entity.id for caption_entity in caption_entities]
        for caption_id, caption, frame_id, timestamp in zip(
            caption_ids, captions, frame_ids, timestamps, strict=True
        ):
            caption_entity = await caption_repo.read(caption_id)

            assert caption_entity.model == model_name
            assert caption_entity.frame_id == frame_id
            assert caption_entity.timestamp == timestamp
            assert caption_entity.caption == caption

        # delete all captions
        for caption_id in caption_ids:
            await caption_repo.delete(caption_id)
            with pytest.raises(NotFoundException):
                await caption_repo.read(caption_id)
