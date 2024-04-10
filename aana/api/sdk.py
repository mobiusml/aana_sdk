import portpicker
import ray
from ray import serve
from ray.serve.handle import RayServeSyncHandle

from aana.api.new_api_generation import Endpoint
from aana.api.new_request_handler import RequestHandler


class AanaSDK:
    def __init__(self, port=8000):
        self.port = port
        self.endpoints = {}
        ray.init()

    def register_deployment(self, name, deployment_instance):
        handle = serve.run(
            deployment_instance.bind(),
            port=self.port,
            name=name,
            route_prefix=f"/{name}",
        )

    def register_endpoint(self, name, path, summary, func):
        endpoint = Endpoint(
            name=name,
            path=path,
            summary=summary,
            func=func,
        )
        self.endpoints[name] = endpoint

    def start_request_handler(self):
        return serve.run(
            RequestHandler.bind(endpoints=self.endpoints.values()),
            port=self.port,
            name="RequestHandler",
            route_prefix="/",
        )


def get_deployment(name):
    return serve.get_app_handle(name)
