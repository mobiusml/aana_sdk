from aana.api.request_handler import RequestHandler
from aana.configs.deployments import deployments


server = RequestHandler.bind(deployments)
