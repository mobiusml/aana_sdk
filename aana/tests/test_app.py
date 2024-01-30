# ruff: noqa: S101, S113
import json

import requests
from ray import serve

from aana.api.api_generation import Endpoint, EndpointOutput


@serve.deployment
class Lowercase:
    """Ray deployment that returns the lowercase version of a text."""

    async def lower(self, text: str) -> dict:
        """Lowercase the text.

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


def test_app(ray_serve_setup):
    """Test the Ray Serve app."""
    handle, port, route_prefix = ray_serve_setup(endpoints, nodes, context)

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
