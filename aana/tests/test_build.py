from mobius_pipeline.exceptions import OutputNotFoundException
import pytest

from aana.api.api_generation import Endpoint, EndpointOutput
from aana.configs.build import get_configuration

nodes = [
    {
        "name": "text",
        "type": "input",
        "inputs": [],
        "outputs": [{"name": "text", "key": "text", "path": "texts.[*].text"}],
    },
    {
        "name": "number",
        "type": "input",
        "inputs": [],
        "outputs": [{"name": "number", "key": "number", "path": "numbers.[*].number"}],
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
    {
        "name": "uppercase",
        "type": "ray_deployment",
        "deployment_name": "Uppercase",
        "method": "upper",
        "inputs": [{"name": "text", "key": "text", "path": "texts.[*].text"}],
        "outputs": [
            {
                "name": "uppercase_text",
                "key": "text",
                "path": "texts.[*].uppercase_text",
            }
        ],
    },
    {
        "name": "capitalize",
        "type": "ray_deployment",
        "deployment_name": "Capitalize",
        "method": "capitalize",
        "inputs": [{"name": "text", "key": "text", "path": "texts.[*].text"}],
        "outputs": [
            {
                "name": "capitalize_text",
                "key": "text",
                "path": "texts.[*].capitalize_text",
            }
        ],
    },
]

# don't define the deployment for "Capitalize" to test if get_configuration raises an error
deployments = {"Lowercase": "Lowercase", "Uppercase": "Uppercase"}

endpoints = {
    "lowercase": [
        Endpoint(
            name="lowercase",
            path="/lowercase",
            summary="Lowercase text",
            outputs=[EndpointOutput(name="lowercase_text", output="lowercase_text")],
        )
    ],
    "uppercase": [
        Endpoint(
            name="uppercase",
            path="/uppercase",
            summary="Uppercase text",
            outputs=[EndpointOutput(name="uppercase_text", output="uppercase_text")],
        )
    ],
    "both": [
        Endpoint(
            name="lowercase",
            path="/lowercase",
            summary="Lowercase text",
            outputs=[EndpointOutput(name="lowercase_text", output="lowercase_text")],
        ),
        Endpoint(
            name="uppercase",
            path="/uppercase",
            summary="Uppercase text",
            outputs=[EndpointOutput(name="uppercase_text", output="uppercase_text")],
        ),
    ],
    "non_existent": [
        Endpoint(
            name="non_existent",
            path="/non_existent",
            summary="Non existent endpoint",
            outputs=[EndpointOutput(name="non_existent", output="non_existent")],
        )
    ],
    "capitalize": [
        Endpoint(
            name="capitalize",
            path="/capitalize",
            summary="Capitalize text",
            outputs=[EndpointOutput(name="capitalize_text", output="capitalize_text")],
        )
    ],
}


@pytest.mark.parametrize(
    "target, expected_nodes, expected_deployments",
    [
        ("lowercase", ["text", "lowercase"], {"Lowercase": "Lowercase"}),
        ("uppercase", ["text", "uppercase"], {"Uppercase": "Uppercase"}),
        (
            "both",
            ["text", "lowercase", "uppercase"],
            {"Lowercase": "Lowercase", "Uppercase": "Uppercase"},
        ),
    ],
)
def test_get_configuration_success(target, expected_nodes, expected_deployments):
    """
    Test if get_configuration returns the correct configuration for various targets
    """
    configuration = get_configuration(target, endpoints, nodes, deployments)
    assert configuration["endpoints"] == endpoints[target]

    node_names = [node["name"] for node in configuration["nodes"]]
    for expected_node in expected_nodes:
        assert expected_node in node_names

    assert configuration["deployments"] == expected_deployments


def test_get_configuration_invalid_target():
    """
    Test if get_configuration raises an error if the target is invalid
    """

    with pytest.raises(ValueError):
        get_configuration("invalid_target", endpoints, nodes, deployments)


def test_get_configuration_non_existent_output():
    """
    Test if get_configuration raises an error
    if one of the target endpoints has a non-existent output.
    """

    with pytest.raises(OutputNotFoundException):
        get_configuration("non_existent", endpoints, nodes, deployments)


def test_get_configuration_not_defined_deployment():
    """
    Test if get_configuration raises an error
    if one of the target nodes uses a deployment that is not defined.
    """

    with pytest.raises(ValueError):
        get_configuration("capitalize", endpoints, nodes, deployments)
