# from aana.core.models.captions import Caption


# caption_repo.save_all(
#     model_name="whisper",
#     captions=[
#         Caption("This is a caption"),
#         Caption("This is another caption"),
#         Caption("This is a third caption"),
#     ],
#     timestamps=[0.1, 0.2, 0.3],
#     frame_ids=[0, 1, 2],
# )

import uuid
from random import random

import pytest

from aana.core.models.captions import Caption
from aana.exceptions.db import NotFoundException
from aana.storage.repository.caption import CaptionRepository


@pytest.fixture(scope="function")
def dummy_caption():
    """Creates a dummy caption for testing."""
    caption = Caption(f"This is a caption {uuid.uuid4()}")
    frame_id = random.randint(0, 100)
    timestamp = random.random()
    return caption, frame_id, timestamp


def test_save_caption(db_session, dummy_caption):
    """Tests saving a caption."""
    caption, frame_id, timestamp = dummy_caption
    model_name = "blip2"

    caption_repo = CaptionRepository(db_session)
    caption_entity = caption_repo.save(
        model_name=model_name,
        caption=caption,
        frame_id=frame_id,
        timestamp=timestamp,
    )
    caption_id = caption_entity.id

    caption_entity = caption_repo.read(caption_id)
    assert caption_entity.model == model_name
    assert caption_entity.frame_id == frame_id
    assert caption_entity.timestamp == timestamp
    assert caption_entity.caption == caption

    caption_repo.delete(caption_id)
    with pytest.raises(NotFoundException):
        caption_repo.read(caption_id)


def test_save_all_captions(db_session, dummy_caption):
    """Tests saving all captions."""
    captions, frame_ids, timestamps = [], [], []
    for _ in range(3):
        caption, frame_id, timestamp = dummy_caption
        captions.append(caption)
        frame_ids.append(frame_id)
        timestamps.append(timestamp)
    model_name = "blip2"

    caption_repo = CaptionRepository(db_session)
    caption_entities = caption_repo.save_all(
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
        caption_entity = caption_repo.read(caption_id)

        assert caption_entity.model == model_name
        assert caption_entity.frame_id == frame_id
        assert caption_entity.timestamp == timestamp
        assert caption_entity.caption == caption

    # delete all captions
    for caption_id in caption_ids:
        caption_repo.delete(caption_id)
        with pytest.raises(NotFoundException):
            caption_repo.read(caption_id)
