import json
from sys import prefix
import time
from typing import AsyncGenerator
import pytest
import ray
from ray import serve
import requests

from aana.api.api_generation import Endpoint
from aana.api.request_handler import RequestHandler


@serve.deployment
class Lowercase:
    """Lowercase class is a Ray Serve deployment class that takes a text
    and returns the lowercase version of it.
    """

    async def lower_stream(self, text: str) -> AsyncGenerator[dict, None]:
        """
        Lowercase the text and yield the lowercase text in chunks of 4 characters.

        Args:
            text (str): The text to lowercase

        Yields:
            dict: The lowercase text in chunks of 4 characters
        """

        chunk_size = 4
        for i in range(0, len(text), chunk_size):
            t = text[i : i + chunk_size]
            yield {"text": t.lower()}


nodes = [
    {
        "name": "text",
        "type": "input",
        "inputs": [],
        "outputs": [{"name": "text", "key": "text", "path": "text"}],
    },
    {
        "name": "lowercase",
        "type": "ray_deployment",
        "deployment_name": "Lowercase",
        "data_type": "generator",
        "generator_path": "text",
        "method": "lower_stream",
        "inputs": [{"name": "text", "key": "text", "path": "text"}],
        "outputs": [
            {
                "name": "lowercase_text",
                "key": "text",
                "path": "lowercase_text",
            }
        ],
    },
]

context = {
    "deployments": {
        "Lowercase": Lowercase.bind(),
    }
}


endpoints = [
    Endpoint(
        name="lowercase",
        path="/lowercase",
        summary="Lowercase text",
        outputs=["lowercase_text"],
        streaming=True,
    )
]


@pytest.fixture(scope="session")
def ray_setup(request):
    """
    Setup the Ray environment and serve the endpoints.

    Returns:
        tuple: A tuple containing the handle to the Ray Serve app
        and the port on which the app is running.
    """
    ray.init(ignore_reinit_error=True)
    server = RequestHandler.bind(endpoints, nodes, context)
    port = 34422
    test_name = request.node.name
    route_prefix = f"/{test_name}"
    handle = serve.run(server, port=port, name=test_name, route_prefix=route_prefix)
    return handle, port, route_prefix


def test_app_streaming(ray_setup):
    """
    Test the Ray Serve app with streaming enabled.
    """

    handle, port, route_prefix = ray_setup

    # Check that the server is ready
    response = requests.get(f"http://localhost:{port}{route_prefix}/api/ready")
    assert response.status_code == 200
    assert response.json() == {"ready": True}

    # Test lowercase endpoint
    text = "Hello World, this is a test."
    data = {"text": text}
    response = requests.post(
        f"http://localhost:{port}{route_prefix}/lowercase",
        data={"body": json.dumps(data)},
        stream=True,
    )
    assert response.status_code == 200

    lowercase_text = ""
    offset = 0
    for chunk in response.iter_content(chunk_size=None):
        json_data = json.loads(chunk)
        lowercase_text_chunk = json_data["lowercase_text"]
        lowercase_text += lowercase_text_chunk

        chunk_size = len(lowercase_text_chunk)
        expected_chunk = text[offset : offset + chunk_size]

        assert lowercase_text_chunk == expected_chunk.lower()

        offset += chunk_size

    assert lowercase_text == text.lower()
