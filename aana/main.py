from aana.api.request_handler import RequestHandler
from aana.configs.endpoints import endpoints
from aana.configs.deployments import deployments
from aana.configs.pipeline import nodes as pipeline_nodes

# TODO: add build system to only serve the deployment if it's needed

context = {
    "deployments": {name: deployment.bind() for name, deployment in deployments.items()}
}

server = RequestHandler.bind(endpoints, pipeline_nodes, context)
