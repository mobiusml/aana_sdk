# ruff: noqa: S101, NPY002 TODO
import numpy as np
import pytest

from aana.processors.batch import BatchProcessor

NUM_IMAGES = 5
IMAGE_SIZE = 100
FEATURE_SIZE = 10


@pytest.fixture
def images():
    """Gets random-data "images" for use in tests."""
    return np.array([np.random.rand(IMAGE_SIZE, IMAGE_SIZE) for _ in range(NUM_IMAGES)])


@pytest.fixture
def texts():
    """Gets random-data "texts" for use in tests."""
    return [f"text{i}" for i in range(NUM_IMAGES)]


@pytest.fixture
def features():
    """Gets random data "feature vectors" for use in tests."""
    return np.random.rand(NUM_IMAGES, FEATURE_SIZE)


@pytest.fixture
def request_batch(images, texts, features):
    """Sample request batch for use in tests."""
    return {"images": images, "texts": texts, "features": features}


@pytest.fixture
def process_batch():
    """Dummy processing function that just returns the batch."""
    return lambda batch: batch


def test_batch_iterator(request_batch, process_batch):
    """Test batch iterator."""
    batch_size = 2
    processor = BatchProcessor(
        process_batch=process_batch, batch_size=batch_size, num_threads=2
    )

    batches = list(processor.batch_iterator(request_batch))
    # We expect 3 batches with a batch_size of 2 for 5 items
    assert len(batches) == NUM_IMAGES // batch_size + 1
    assert all(
        len(batch["texts"]) == batch_size for batch in batches[:-1]
    )  # All but the last should have batch_size items
    assert len(batches[-1]["texts"]) == 1  # Last batch should have the remaining item


@pytest.mark.asyncio
async def test_process_batches(request_batch, process_batch):
    """Test processing of batches."""
    batch_size = 2
    processor = BatchProcessor(
        process_batch=process_batch, batch_size=batch_size, num_threads=2
    )

    result = await processor.process(request_batch)
    # Ensure all texts are processed
    assert len(result["texts"]) == NUM_IMAGES
    # Check if images are concatenated properly
    assert result["images"].shape == (NUM_IMAGES, IMAGE_SIZE, IMAGE_SIZE)
    # Check if features are concatenated properly
    assert result["features"].shape == (NUM_IMAGES, FEATURE_SIZE)


def test_merge_outputs(request_batch, process_batch):
    """Test merging of outputs from multiple batches.

    images and features should be concatenated into numpy arrays because they are numpy arrays.
    texts should be concatenated because they are lists.
    """
    batch_size = 2
    processor = BatchProcessor(
        process_batch=process_batch, batch_size=batch_size, num_threads=2
    )

    # Assume the processor has already batched and processed the data
    processed_batches = [
        {
            "images": request_batch["images"][:2],
            "texts": request_batch["texts"][:2],
            "features": request_batch["features"][:2],
        },
        {
            "images": request_batch["images"][2:4],
            "texts": request_batch["texts"][2:4],
            "features": request_batch["features"][2:4],
        },
        {
            "images": request_batch["images"][4:],
            "texts": request_batch["texts"][4:],
            "features": request_batch["features"][4:],
        },
    ]

    merged_output = processor.merge_outputs(processed_batches)
    assert merged_output["texts"] == request_batch["texts"]
    assert np.array_equal(merged_output["images"], request_batch["images"])
    assert np.array_equal(merged_output["features"], request_batch["features"])
