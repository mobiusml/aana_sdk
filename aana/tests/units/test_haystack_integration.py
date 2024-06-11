# ruff: noqa: S101, S113
import json
from typing import TypedDict

import requests
from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from aana.api.api_generation import Endpoint
from aana.deployments.haystack_component_deployment import (
    HaystackComponentDeployment,
    HaystackComponentDeploymentConfig,
    RemoteHaystackComponent,
)


class HaystackTestEndpointOutput(TypedDict):
    """The output of the Haystack test endpoint."""

    response: str


class HaystackTestEndpoint(Endpoint):
    """A test endpoint for Haystack Integration."""

    async def initialize(self):
        """Initialize the endpoint."""
        document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

        documents = [
            Document(content="My name is Wolfgang and I live in Berlin"),
            Document(content="I saw a black horse running"),
            Document(content="Germany has many big cities"),
        ]

        document_embedder = RemoteHaystackComponent("document_embedder_deployment")
        document_embedder.warm_up()
        documents_with_embeddings = document_embedder.run(documents=documents)[
            "documents"
        ]
        document_store.write_documents(documents_with_embeddings)

        text_embedder = RemoteHaystackComponent("text_embedder_deployment")
        text_embedder.warm_up()

        self.query_pipeline = Pipeline()
        self.query_pipeline.add_component("text_embedder", text_embedder)
        self.query_pipeline.add_component(
            "retriever", InMemoryEmbeddingRetriever(document_store=document_store)
        )
        self.query_pipeline.connect(
            "text_embedder.embedding", "retriever.query_embedding"
        )
        await super().initialize()

    async def run(self, query: str) -> HaystackTestEndpointOutput:
        """Run the test endpoint for Haystack Integration."""
        result = self.query_pipeline.run({"text_embedder": {"text": query}})
        return {"response": result["retriever"]["documents"][0].content}


text_embedder_deployment = HaystackComponentDeployment.options(
    num_replicas=1,
    max_concurrent_queries=1000,
    user_config=HaystackComponentDeploymentConfig(
        component="haystack.components.embedders.SentenceTransformersTextEmbedder",
        params={"model": "sentence-transformers/all-mpnet-base-v2"},
    ).model_dump(),
)

document_embedder_deployment = HaystackComponentDeployment.options(
    num_replicas=1,
    max_concurrent_queries=1000,
    user_config=HaystackComponentDeploymentConfig(
        component="haystack.components.embedders.SentenceTransformersDocumentEmbedder",
        params={"model": "sentence-transformers/all-mpnet-base-v2"},
    ).model_dump(),
)


deployments = [
    {
        "name": "text_embedder_deployment",
        "instance": text_embedder_deployment,
    },
    {
        "name": "document_embedder_deployment",
        "instance": document_embedder_deployment,
    },
]

endpoints = [
    {
        "name": "haystack_test_endpoint",
        "path": "/query",
        "summary": "A test endpoint for Haystack",
        "endpoint_cls": HaystackTestEndpoint,
    }
]


def test_haystack_integration(app_setup):
    """Test Haystack integration."""
    aana_app = app_setup(deployments, endpoints)

    port = aana_app.port

    data = {"query": "Who lives in Berlin?"}
    response = requests.post(
        f"http://localhost:{port}/query",
        data={"body": json.dumps(data)},
    )
    assert response.status_code == 200
    assert response.json() == {"response": "My name is Wolfgang and I live in Berlin"}

    data = {"query": "What is the interesting fact about Germany?"}
    response = requests.post(
        f"http://localhost:{port}/query",
        data={"body": json.dumps(data)},
    )

    assert response.status_code == 200
    assert response.json() == {"response": "Germany has many big cities"}
