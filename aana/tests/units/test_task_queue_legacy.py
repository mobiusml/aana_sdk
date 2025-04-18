# ruff: noqa: S101, S113
import json
import time
from collections.abc import AsyncGenerator
from typing import Annotated, TypedDict

import requests
from pydantic import Field
from ray import serve

from aana.api.api_generation import Endpoint
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.base_deployment import BaseDeployment


@serve.deployment
class Lowercase(BaseDeployment):
    """Ray deployment that returns the lowercase version of a text."""

    async def lower(self, text: str) -> dict:
        """Lowercase the text.

        Args:
            text (str): The text to lowercase

        Returns:
            dict: The lowercase text
        """
        return {"text": [t.lower() for t in text]}


TextList = Annotated[list[str], Field(description="List of text to lowercase.")]


class LowercaseEndpointOutput(TypedDict):
    """The output of the lowercase endpoint."""

    text: list[str]


class LowercaseEndpoint(Endpoint):
    """Lowercase endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        self.lowercase_handle = await AanaDeploymentHandle.create(
            "lowercase_deployment"
        )
        await super().initialize()

    async def run(self, text: TextList) -> LowercaseEndpointOutput:
        """Lowercase the text.

        Args:
            text (TextList): The list of text to lowercase

        Returns:
            LowercaseEndpointOutput: The lowercase texts
        """
        lowercase_output = await self.lowercase_handle.lower(text=text)
        return {"text": lowercase_output["text"]}


class LowercaseSteamEndpoint(Endpoint):
    """Lowercase endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        self.lowercase_handle = await AanaDeploymentHandle.create(
            "lowercase_deployment"
        )
        await super().initialize()

    async def run(
        self, text: TextList
    ) -> AsyncGenerator[LowercaseEndpointOutput, None]:
        """Lowercase the text.

        Args:
            text (TextList): The list of text to lowercase

        Returns:
            LowercaseEndpointOutput: The lowercase texts
        """
        lowercase_output = await self.lowercase_handle.lower(text=text)
        for t in lowercase_output["text"]:
            yield {"text": t}


deployments = [
    {
        "name": "lowercase_deployment",
        "instance": Lowercase,
    },
]


endpoints = [
    {
        "name": "lowercase",
        "path": "/lowercase",
        "summary": "Lowercase text",
        "endpoint_cls": LowercaseEndpoint,
    },
    {
        "name": "lowercase_stream",
        "path": "/lowercase_stream",
        "summary": "Lowercase text",
        "endpoint_cls": LowercaseSteamEndpoint,
    },
]


def test_task_queue(create_app):  # noqa: C901
    """Test the Ray Serve app."""
    aana_app = create_app(deployments, endpoints)

    port = aana_app.port
    # Check that the server is ready
    response = requests.get(f"http://localhost:{port}/api/ready")
    assert response.status_code == 200
    assert response.json() == {"ready": True}

    # Test lowercase endpoint
    data = {"text": ["Hello World!", "This is a test."]}
    response = requests.post(
        f"http://localhost:{port}/lowercase",
        data={"body": json.dumps(data)},
    )
    assert response.status_code == 200
    lowercase_text = response.json().get("text")
    assert lowercase_text == ["hello world!", "this is a test."]

    # Defer endpoint execution
    response = requests.post(
        f"http://localhost:{port}/lowercase?defer=True",
        data={"body": json.dumps(data)},
    )
    assert response.status_code == 200
    task_id = response.json().get("task_id")

    # Check the task status with timeout of 10 seconds
    start_time = time.time()
    while time.time() - start_time < 10:
        response = requests.get(f"http://localhost:{port}/tasks/get/{task_id}")
        task_status = response.json().get("status")
        result = response.json().get("result")
        if task_status == "completed":
            break
        time.sleep(0.1)

    assert task_status == "completed", response.text
    assert result == {"text": ["hello world!", "this is a test."]}

    # Delete the task
    response = requests.get(f"http://localhost:{port}/tasks/delete/{task_id}")
    assert response.status_code == 200
    assert response.json().get("task_id") == task_id

    # Check that the task is deleted
    response = requests.get(f"http://localhost:{port}/tasks/get/{task_id}")
    assert response.status_code == 404
    assert response.json().get("error") == "NotFoundException"

    # Check non-existent task
    task_id = "d1b1b1b1-1b1b-1b1b-1b1b-1b1b1b1b1b1b"
    response = requests.get(f"http://localhost:{port}/tasks/get/{task_id}")
    assert response.status_code == 404
    assert response.json().get("error") == "NotFoundException"

    # Test lowercase streaming endpoint
    data = {"text": ["Hello World!", "This is a test."]}
    response = requests.post(
        f"http://localhost:{port}/lowercase_stream",
        data={"body": json.dumps(data)},
        stream=True,
    )
    assert response.status_code == 200

    for i, chunk in enumerate(response.iter_content(chunk_size=None)):
        json_data = json.loads(chunk)
        assert "text" in json_data
        assert json_data["text"] == lowercase_text[i]

    # Test task queue with streaming endpoint
    data = {"text": ["Hello World!", "This is a test."]}
    response = requests.post(
        f"http://localhost:{port}/lowercase_stream?defer=True",
        data={"body": json.dumps(data)},
    )
    assert response.status_code == 200
    task_id = response.json().get("task_id")

    # Check the task status with timeout of 10 seconds
    start_time = time.time()
    while time.time() - start_time < 10:
        response = requests.get(f"http://localhost:{port}/tasks/get/{task_id}")
        task_status = response.json().get("status")
        result = response.json().get("result")
        if task_status == "completed":
            break
        time.sleep(0.1)

    assert task_status == "completed", response.text
    assert [chunk["text"] for chunk in result] == lowercase_text

    # Send 30 tasks to the task queue
    task_ids = []
    for i in range(30):
        data = {"text": [f"Task {i}"]}
        response = requests.post(
            f"http://localhost:{port}/lowercase_stream?defer=True",
            data={"body": json.dumps(data)},
        )
        assert response.status_code == 200
        task_ids.append(response.json().get("task_id"))

    # Check the task statusES with timeout of 10 seconds
    start_time = time.time()
    completed_tasks = []
    while time.time() - start_time < 10:
        for task_id in task_ids:
            if task_id in completed_tasks:
                continue
            response = requests.get(f"http://localhost:{port}/tasks/get/{task_id}")
            task_status = response.json().get("status")
            result = response.json().get("result")
            if task_status == "completed":
                completed_tasks.append(task_id)

        if len(completed_tasks) == len(task_ids):
            break
        time.sleep(0.1)

    # Check that all tasks are completed
    for task_id in task_ids:
        response = requests.get(f"http://localhost:{port}/tasks/get/{task_id}")
        response = response.json()
        task_status = response.get("status")
        assert task_status == "completed", response
