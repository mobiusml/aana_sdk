from typing import Dict

from mobius_pipeline.node.node_definition import NodeDefinition
from mobius_pipeline.pipeline.output_graph import OutputGraph


def get_configuration(target: str, endpoints, nodes, deployments) -> Dict:
    """
    Returns the configuration for the specified target.

    A target is a set of endpoints that are to be deployed together.

    All the targets are defined in the endpoints.py file.

    The function finds:
        - which endpoints are to be deployed
        - which nodes are to be used in the pipeline
        - which Ray Deployments need to be deployed

    Args:
        target (str): The name of the target to be deployed.
        endpoints (Dict): The dictionary of endpoints.
        nodes (List): The list of nodes.
        deployments (Dict): The dictionary of Ray Deployments.

    Returns:
        Dict: The dictionary with the configuration.
              The configuration contains 3 keys:
                - endpoints
                - nodes
                - deployments
    """

    # Check if target is valid
    if target not in endpoints:
        raise ValueError(
            f"Invalid target: {target}. Valid targets: {', '.join(endpoints.keys())}"
        )

    # Find the endpoints that are to be deployed
    target_endpoints = endpoints[target]

    # Target endpoints require the following outputs
    endpoint_outputs = []
    for endpoint in target_endpoints:
        endpoint_outputs += endpoint.outputs

    # Build the output graph for the whole pipeline
    node_definitions = [NodeDefinition.from_dict(node_dict) for node_dict in nodes]
    outputs_graph = OutputGraph(node_definitions)

    # Find what inputs are required for the endpoint outputs
    inputs = outputs_graph.find_input_nodes(endpoint_outputs)
    # Target outputs are the inputs + subgraph of the pipeline
    # that is required to generate the outputs for endpoints
    target_outputs = inputs + outputs_graph.find_subgraph(inputs, endpoint_outputs)

    # Now we have the target outputs, we can find the nodes that generate them.
    # Find the nodes that generate the target outputs
    target_nodes = []
    for node in nodes:
        node_output_names = [output["name"] for output in node["outputs"]]
        if any([output in node_output_names for output in target_outputs]):
            target_nodes.append(node)

    # Now we have the target nodes, we can find the Ray Deployments that they use.
    # Find the Ray Deployments that are used by the target nodes
    target_deployment_names = set()
    for node in target_nodes:
        if node["type"] == "ray_deployment":
            target_deployment_names.add(node["deployment_name"])

    target_deployments = {}
    for deployment_name in target_deployment_names:
        if deployment_name not in deployments:
            raise ValueError(f"Deployment {deployment_name} is not defined.")
        target_deployments[deployment_name] = deployments[deployment_name]

    return {
        "endpoints": target_endpoints,
        "nodes": target_nodes,
        "deployments": target_deployments,
    }
