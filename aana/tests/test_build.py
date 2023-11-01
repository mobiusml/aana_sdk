from mobius_pipeline.exceptions import OutputNotFoundException
import pytest

from aana.api.api_generation import Endpoint
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
            outputs=["lowercase_text"],
        )
    ],
    "uppercase": [
        Endpoint(
            name="uppercase",
            path="/uppercase",
            summary="Uppercase text",
            outputs=["uppercase_text"],
        )
    ],
    "both": [
        Endpoint(
            name="lowercase",
            path="/lowercase",
            summary="Lowercase text",
            outputs=["lowercase_text"],
        ),
        Endpoint(
            name="uppercase",
            path="/uppercase",
            summary="Uppercase text",
            outputs=["uppercase_text"],
        ),
    ],
    "non_existent": [
        Endpoint(
            name="non_existent",
            path="/non_existent",
            summary="Non existent endpoint",
            outputs=["non_existent"],
        )
    ],
    "capitalize": [
        Endpoint(
            name="capitalize",
            path="/capitalize",
            summary="Capitalize text",
            outputs=["capitalize_text"],
        )
    ],
}


def test_get_configuration_success():
    """
    Test if get_configuration returns the correct configuration
    """

    # Test lowercase target

    configuration = get_configuration("lowercase", endpoints, nodes, deployments)
    assert configuration["endpoints"] == endpoints["lowercase"]
    # nodes should have "text" and "lowercase"
    node_names = [node["name"] for node in configuration["nodes"]]
    assert "text" in node_names
    assert "lowercase" in node_names
    # deployments should have "Lowercase"
    assert configuration["deployments"] == {"Lowercase": "Lowercase"}

    # Test uppercase target

    configuration = get_configuration("uppercase", endpoints, nodes, deployments)
    assert configuration["endpoints"] == endpoints["uppercase"]
    # nodes should have "text" and "uppercase"
    node_names = [node["name"] for node in configuration["nodes"]]
    assert "text" in node_names
    assert "uppercase" in node_names
    # deployments should have "Uppercase"
    assert configuration["deployments"] == {"Uppercase": "Uppercase"}

    # Test both target

    configuration = get_configuration("both", endpoints, nodes, deployments)
    assert configuration["endpoints"] == endpoints["both"]
    # nodes should have "text", "lowercase" and "uppercase"
    node_names = [node["name"] for node in configuration["nodes"]]
    assert "text" in node_names
    assert "lowercase" in node_names
    assert "uppercase" in node_names
    # deployments should have "Lowercase" and "Uppercase"
    assert configuration["deployments"] == {
        "Lowercase": "Lowercase",
        "Uppercase": "Uppercase",
    }


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
