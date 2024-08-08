# ruff: noqa: S101
from typing import Any, TypedDict

import haystack
import haystack.components.preprocessors
import numpy as np
import pytest
from haystack import Document
from haystack.components.preprocessors import TextCleaner
from ray import serve

from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.base_deployment import BaseDeployment
from aana.integrations.haystack.deployment_component import AanaDeploymentComponent


class DummyEmbedderOutput(TypedDict):
    """Output of the dummy embedder."""

    embeddings: np.ndarray


@serve.deployment
class DummyEmbedder(BaseDeployment):
    """Simple deployment to store video data."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration to the deployment and initialize it."""
        pass

    async def embed(self, texts: list[str]) -> DummyEmbedderOutput:
        """Generate a dummy embedding for the texts."""
        return {"embeddings": np.random.default_rng().random((len(texts), 8))}

    async def embed_fail(self, texts: list[str]) -> DummyEmbedderOutput:
        """Fail to generate a dummy embedding for the texts."""
        raise Exception("Dummy exception")  # noqa: TRY002, TRY003


@pytest.fixture(scope="module")
def setup_dummy_embedder_deployment(create_app):
    """Set up the dummy embedder deployment."""
    deployment_name = "dummy_embedder"
    deployment = DummyEmbedder.options(num_replicas=1, user_config={})
    deployments = [
        {
            "name": deployment_name,
            "instance": deployment,
        }
    ]
    endpoints = []

    return deployment_name, create_app(deployments, endpoints)


@pytest.mark.asyncio
async def test_haystack_wrapper(setup_dummy_embedder_deployment):
    """Tests haystack wrapper for deployments."""
    deployment_name, _ = setup_dummy_embedder_deployment
    deployment_handle = await AanaDeploymentHandle.create(deployment_name)
    component = AanaDeploymentComponent(deployment_handle, "embed")
    result = component.run(texts=["Hello, world!", "Roses are red", "Violets are blue"])

    assert "embeddings" in result
    assert isinstance(result["embeddings"], np.ndarray)
    assert result["embeddings"].shape == (3, 8)


@pytest.mark.asyncio
async def test_haystack_pipeline(setup_dummy_embedder_deployment):
    """Tests haystack wrapper in a pipeline."""
    deployment_name, _ = setup_dummy_embedder_deployment

    @haystack.component
    class DocumentCombinerComponent:
        @haystack.component.output_types(documents=list[Document])
        def run(self, texts: list[str], embeddings: np.ndarray):
            documents = [
                Document(content=text, embedding=embedding)
                for text, embedding in zip(texts, embeddings, strict=True)
            ]
            return {"documents": documents}

    embedder_handle = await AanaDeploymentHandle.create(deployment_name)
    embedder = AanaDeploymentComponent(embedder_handle, "embed")

    text_cleaner = TextCleaner(
        convert_to_lowercase=False, remove_punctuation=True, remove_numbers=True
    )

    document_combiner = DocumentCombinerComponent()

    pipeline = haystack.Pipeline()
    pipeline.add_component("text_cleaner", text_cleaner)
    pipeline.add_component("dummy_embedder", embedder)
    pipeline.add_component("document_combiner", document_combiner)
    pipeline.connect("text_cleaner.texts", "dummy_embedder.texts")
    pipeline.connect("dummy_embedder.embeddings", "document_combiner.embeddings")
    pipeline.connect("text_cleaner.texts", "document_combiner.texts")

    texts = ["Hello, world!", "Roses are red", "Violets are blue"]

    output = pipeline.run({"text_cleaner": {"texts": texts}})
    # {'document_combiner':
    #   {'documents':
    #       [Document(id=489009ee9ce0fd7f2aa38658ff7885fe8c2ea70fab5a711b14ee9d5eb65b2843, content: 'Hello world', embedding: vector of size 8),
    #        Document(id=489009ee9ce0fd7f2aa38658ff7885fe8c2ea70fab5a711b14ee9d5eb65b2843, content: 'Roses red', embedding: vector of size 8),
    #        Document(id=489009ee9ce0fd7f2aa38658ff7885fe8c2ea70fab5a711b14ee9d5eb65b2843, content: 'Violets blue', embedding: vector of size 8)
    #       ]
    #   }
    # }

    assert "document_combiner" in output
    assert "documents" in output["document_combiner"]
    assert len(output["document_combiner"]["documents"]) == 3
    assert all(
        isinstance(doc, Document) for doc in output["document_combiner"]["documents"]
    )


@pytest.mark.asyncio
async def test_haystack_wrapper_fails(setup_dummy_embedder_deployment):
    """Tests that haystack wrapper raises if method_name is missing."""
    deployment_name, _ = setup_dummy_embedder_deployment
    missing_method_name = "does_not_exist"
    deployment_handle = await AanaDeploymentHandle.create(deployment_name)
    with pytest.raises(AttributeError):
        AanaDeploymentComponent(deployment_handle, missing_method_name)

    failing_method_name = "embed_fail"
    component = AanaDeploymentComponent(deployment_handle, failing_method_name)
    with pytest.raises(Exception):  # noqa: B017
        component.run(texts=["Hello, world!", "Roses are red", "Violets are blue"])
