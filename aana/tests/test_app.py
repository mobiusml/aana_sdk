import json
import pytest
import ray
from ray import serve
import requests

from aana.api.api_generation import Endpoint, EndpointOutput
from aana.api.request_handler import RequestHandler


@serve.deployment
class Lowercase:
    """Lowercase class is a Ray Serve deployment class that takes a text
    and returns the lowercase version of it.
    """

    async def lower(self, text: str) -> dict:
        """
        Lowercase the text.

        Args:
            text (str): The text to lowercase

        Returns:
            dict: The lowercase text
        """
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
        outputs=[EndpointOutput(name="text", output="lowercase_text")],
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


def test_app(ray_setup):
    """
    Test the Ray Serve app.
    """
    handle, port, route_prefix = ray_setup

    # Check that the server is ready
    response = requests.get(f"http://localhost:{port}{route_prefix}/api/ready")
    assert response.status_code == 200
    assert response.json() == {"ready": True}

    # Test lowercase endpoint
    data = {"text": ["Hello World!", "This is a test."]}
    response = requests.post(
        f"http://localhost:{port}{route_prefix}/lowercase",
        data={"body": json.dumps(data)},
    )
    assert response.status_code == 200
    lowercase_text = response.json().get("text")
    assert lowercase_text == ["hello world!", "this is a test."]
