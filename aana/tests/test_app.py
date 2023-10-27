import json
import random
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

    async def lower(self, text):
        return {"text": [t.lower() for t in text]}


nodes = [
    {
        "name": "text",
        "type": "input",
        "inputs": [],
        "outputs": [{"name": "text", "key": "text", "path": "texts.[*].text"}],
    },
    {
        "name": "lowercase",
        "type": "ray_deployment",
        "deployment_name": "Lowercase",
        "method": "lower",
        "inputs": [{"name": "text", "key": "text", "path": "texts.[*].text"}],
        "outputs": [
            {
                "name": "lowercase_text",
                "key": "text",
                "path": "texts.[*].lowercase_text",
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
    )
]


@pytest.fixture(scope="session")
def ray_setup():
    # Setup ray environment and serve
    ray.init(ignore_reinit_error=True)
    server = RequestHandler.bind(endpoints, nodes, context)
    # random port from 30000 to 40000
    port = random.randint(30000, 40000)
    handle = serve.run(server, port=port)
    return handle, port


def test_app(ray_setup):
    handle, port = ray_setup

    # Check that the server is ready
    response = requests.get(f"http://localhost:{port}/api/ready")
    assert response.status_code == 200
    assert response.json() == {"ready": True}

    # Test lowercase endpoint
    data = {"text": ["Hello World!", "This is a test."]}
    response = requests.post(
        f"http://localhost:{port}/lowercase", data={"body": json.dumps(data)}
    )
    assert response.status_code == 200
    lowercase_text = response.json().get("lowercase_text")
    assert lowercase_text == ["hello world!", "this is a test."]
