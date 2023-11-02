import pytest
import numpy as np

from aana.utils.batch_processor import BatchProcessor


@pytest.fixture
def images():
    return np.array([np.random.rand(100, 100) for _ in range(5)])  # 5 images


@pytest.fixture
def texts():
    return ["text1", "text2", "text3", "text4", "text5"]


@pytest.fixture
def features():
    return np.random.rand(5, 10)  # 5 features of size 10


@pytest.fixture
def request_batch(images, texts, features):
    return {"images": images, "texts": texts, "features": features}


@pytest.fixture
def process_batch():
    # Dummy processing function that just returns the batch
    return lambda batch: batch


def test_batch_iterator(request_batch, process_batch):
    """
    Test batch iterator.
    """
    batch_size = 2
    processor = BatchProcessor(
        process_batch=process_batch, batch_size=batch_size, num_threads=2
    )

    batches = list(processor.batch_iterator(request_batch))
    assert len(batches) == 3  # We expect 3 batches with a batch_size of 2 for 5 items
    assert all(
        len(batch["texts"]) == batch_size for batch in batches[:-1]
    )  # All but the last should have batch_size items
    assert len(batches[-1]["texts"]) == 1  # Last batch should have the remaining item


@pytest.mark.asyncio
async def test_process_batches(request_batch, process_batch):
    """
    Test processing of batches.
    """
    batch_size = 2
    processor = BatchProcessor(
        process_batch=process_batch, batch_size=batch_size, num_threads=2
    )

    result = await processor.process(request_batch)
    # Ensure all texts are processed
    assert len(result["texts"]) == 5
    # Check if images are concatenated properly
    assert result["images"].shape == (5, 100, 100)
    # Check if features are concatenated properly
    assert result["features"].shape == (5, 10)


def test_merge_outputs(request_batch, process_batch):
    """
    Test merging of outputs from multiple batches.

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
