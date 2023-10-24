from aana.api.request_handler import RequestHandler
from aana.configs.deployments import deployments

# TODO: add build system to only serve the deployment if it's needed
binded_deployments = {}
for name, deployment in deployments.items():
    binded_deployments[name] = deployment.bind()

server = RequestHandler.bind(binded_deployments)
