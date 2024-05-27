from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from ray import serve
from sentence_transformers import SentenceTransformer
from typing_extensions import TypedDict

from aana.deployments.base_deployment import BaseDeployment, test_cache
from aana.exceptions.runtime import InferenceException
from aana.processors.batch import BatchProcessor


class SentenceTransformerConfig(BaseModel):
    """The configuration for the BLIP2 deployment with HuggingFace models.

    Attributes:
        model (str): the model ID on HuggingFace
        batch_size (int): the batch size (optional, default: 1)
        num_processing_threads (int): the number of processing threads (optional, default: 1)
    """

    model: str
    batch_size: int = Field(default=1)
    num_processing_threads: int = Field(default=1)


class SentenceTransformerOutput(TypedDict):
    """The output of the sentence transformer model.

    Attributes:
        embedding (np.ndarray): the embedding
    """

    embedding: np.ndarray


@serve.deployment
class SentenceTransformerDeployment(BaseDeployment):
    """Deployment to serve sentence transformer models."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        It loads the model from HuggingFace.

        The configuration should conform to the SentenceTransformerConfig schema.
        """
        config_obj = SentenceTransformerConfig(**config)

        # Create the batch processor to split the requests into batches
        # and process them in parallel
        self.batch_size = config_obj.batch_size
        self.num_processing_threads = config_obj.num_processing_threads
        # The actual inference is done in _generate()
        # We use lambda because BatchProcessor expects dict as input
        # and we use **kwargs to unpack the dict into named arguments for _generate()
        self.batch_processor = BatchProcessor(
            process_batch=lambda request: self._generate(**request),
            batch_size=self.batch_size,
            num_threads=self.num_processing_threads,
        )

        self.model_id = config_obj.model

        self.model = SentenceTransformer(self.model_id)

    @test_cache
    async def embed_batch(self, **kwargs) -> np.ndarray:
        """Embed the given sentences.

        Args:
            sentences (List[str]): the sentences
            **kwargs (dict[str, Any]): keywordarguments to pass to the
                batch processor.

        Returns:
            np.ndarray: the embeddings for the sentences

        Raises:
            InferenceException: if the inference fails
        """
        # Call the batch processor to process the requests
        # The actual inference is done in _generate()
        return await self.batch_processor.process(kwargs)

    def _generate(self, sentences: list[str]) -> SentenceTransformerOutput:
        """Generate the embeddings for the given sentences.

        This method is called by the batch processor.

        Args:
            sentences (list[str]): the sentences

        Returns:
            SentenceTransformerOutput: the dictionary with one key "embedding"

        Raises:
            InferenceException: if the inference fails
        """
        try:
            embeddings = self.model.encode(sentences)
            return SentenceTransformerOutput(embedding=embeddings)
        except Exception as e:
            raise InferenceException(self.model_id) from e
