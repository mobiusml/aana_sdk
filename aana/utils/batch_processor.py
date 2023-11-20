import asyncio
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np


class BatchProcessor:
    """Class for parallel processing data in chunks.

    The BatchProcessor class encapsulates the logic required to take a large collection of data,
    split it into manageable batches, process these batches in parallel, and then combine the results
    into a single cohesive output.

    Batching works by iterating through the input request, which is a dictionary where each key maps
    to a list-like collection of data. The class splits each collection into sublists of length up
    to `batch_size`, ensuring that corresponding elements across the collections remain grouped
    together in their respective batches.

    Merging takes the output from each processed batch, which is also a dictionary structure, and
    combines these into a single dictionary. Lists are extended, numpy arrays are concatenated, and
    dictionaries are updated. If a new data type is encountered, an error is raised prompting the
    implementer to specify how these types should be merged.

    This class is particularly useful for batching of requests to a machine learning model.

    The thread pool for parallel processing is managed internally and is shut down automatically when
    the BatchProcessor instance is garbage collected.

    Attributes:
        process_batch (Callable): A function to process each batch.
        batch_size (int): The size of each batch to be processed.
        num_threads (int): The number of threads in the thread pool for parallel processing.
    """

    def __init__(self, process_batch: Callable, batch_size: int, num_threads: int):
        """Constructor.

        Args:
            process_batch (Callable): Function that processes each batch.
            batch_size (int): Size of the batches.
            num_threads (int): Number of threads in the pool.
        """
        self.process_batch = process_batch
        self.batch_size = batch_size
        self.pool = ThreadPoolExecutor(num_threads)

    def __del__(self):
        """Destructor that shuts down the thread pool when the instance is destroyed."""
        self.pool.shutdown()

    def batch_iterator(self, request: dict[str, Any]) -> Iterator[dict[str, list[Any]]]:
        """Converts request into an iterator of batches.

        Iterates over the input request, breaking it into smaller batches for processing.
        Each batch is a dictionary with the same keys as the input request, but the values
        are sublists containing only the elements for that batch.

        Example:
            request = {
                'images': [img1, img2, img3, img4, img5],
                'texts': ['text1', 'text2', 'text3', 'text4', 'text5']
            }
            # Assuming a batch size of 2, this iterator would yield:
            # 1st iteration: {'images': [img1, img2], 'texts': ['text1', 'text2']}
            # 2nd iteration: {'images': [img3, img4], 'texts': ['text3', 'text4']}
            # 3rd iteration: {'images': [img5], 'texts': ['text5']}

        Args:
            request (dict[str, list[Any]]): The request data to split into batches.

        Yields:
            Iterator[dict[str, list[Any]]]: An iterator over the batched requests.
        """
        lengths = [len(value) for value in request.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All inputs must have the same length")  # noqa: TRY003

        total_batches = (max(lengths) + self.batch_size - 1) // self.batch_size
        for i in range(total_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            yield {key: value[start:end] for key, value in request.items()}

    async def process(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process a request.

        Splits the input request into batches, processes each batch in parallel, and then merges
        the results into a single dictionary.

        Args:
            request (Dict[str, Any]): The request data to process.

        Returns:
            Dict[str, Any]: The merged results from processing all batches.
        """
        loop = asyncio.get_running_loop()
        futures = [
            loop.run_in_executor(self.pool, self.process_batch, batch)
            for batch in self.batch_iterator(request)
        ]
        outputs = await asyncio.gather(*futures)
        return self.merge_outputs(outputs)

    def merge_outputs(self, outputs: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge output.

        Combine processed batch outputs into a single dictionary. It handles various data types
        by extending lists, updating dictionaries, and concatenating numpy arrays.

        Example:
            outputs = [
                {'images': [processed_img1, processed_img2], 'labels': ['cat', 'dog']},
                {'images': [processed_img3, processed_img4], 'labels': ['bird', 'mouse']},
                {'images': [processed_img5], 'labels': ['fish']}
            ]
            # The merged result would be:
            # {
            #     'images': [processed_img1, processed_img2, processed_img3, processed_img4, processed_img5],
            #     'labels': ['cat', 'dog', 'bird', 'mouse', 'fish']
            # }

        Args:
            outputs (list[dict[str, Any]]): List of outputs from the processed batches.

        Returns:
            dict[str, Any]: The merged result.
        """
        merged_output = {}
        for output in outputs:
            for key, value in output.items():
                if key not in merged_output:
                    merged_output[key] = value
                else:
                    if isinstance(value, list):
                        merged_output[key].extend(value)
                    elif isinstance(value, dict):
                        merged_output[key].update(value)
                    elif isinstance(value, np.ndarray):
                        if key in merged_output:
                            merged_output[key] = np.concatenate(
                                (merged_output[key], value)
                            )
                        else:
                            merged_output[key] = value
                    else:
                        raise NotImplementedError(
                            "Merging of this data type is not implemented"
                        )
        return merged_output
