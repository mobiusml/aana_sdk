import sys
import time
import traceback
from collections.abc import Callable
from typing import Type

import ray
from ray import serve
from ray.serve.deployment import Deployment

from aana.api.api_generation import Endpoint
from aana.api.request_handler import RequestHandler
from aana.configs.db import run_alembic_migrations
from aana.configs.settings import settings as aana_settings


class AanaSDK:
    """Aana SDK to deploy and manage Aana deployments and endpoints."""

    def __init__(self, port: int = 8000):
        """Aana SDK to deploy and manage Aana deployments and endpoints.

        Args:
            port (int, optional): The port to run the Aana server on. Defaults to 8000.
        """
        self.port: int = port
        self.endpoints: dict[str, Endpoint] = {}

        run_alembic_migrations(aana_settings)

        ray.init(ignore_reinit_error=True, log_to_driver=False)

    def register_deployment(
        self, name: str, deployment_instance: Deployment, blocking: bool = False
    ):
        """Register a deployment.

        Args:
            name (str): The name of the deployment.
            deployment_instance (Deployment): The instance of the deployment to be registered.
            blocking (bool, optional): If True, the function will block until deployment is complete. Defaults to False.
        """
        serve.run(
            deployment_instance.bind(),
            port=self.port,
            name=name,
            route_prefix=f"/{name}",
            _blocking=blocking,
        )

    def unregister_deployment(self, name: str):
        """Unregister a deployment.

        Args:
            name (str): The name of the deployment to be unregistered.
        """
        serve.delete(name)

    def register_endpoint(
        self, name: str, path: str, summary: str, endpoint_cls: type[Endpoint]
    ):
        """Register an endpoint.

        Args:
            name (str): The name of the endpoint.
            path (str): The path of the endpoint.
            summary (str): The summary of the endpoint.
            endpoint_cls (Type[Endpoint]): The class of the endpoint.
        """
        endpoint = endpoint_cls(
            name=name,
            path=path,
            summary=summary,
        )
        self.endpoints[name] = endpoint

    def unregister_endpoint(self, name: str):
        """Unregister an endpoint.

        Args:
            name (str): The name of the endpoint to be unregistered.
        """
        if name in self.endpoints:
            del self.endpoints[name]

    def deploy(self, blocking: bool = False):
        """Deploy the registered deployments and endpoints.

        Args:
            blocking (bool, optional): If True, the function will block until interrupted. Defaults to False.
        """
        try:
            serve.run(
                RequestHandler.options(num_replicas=aana_settings.num_workers).bind(
                    endpoints=self.endpoints.values()
                ),
                port=self.port,
                name="RequestHandler",
                route_prefix="/",
                _blocking=blocking,
            )
            print("Deployed successfully.")
            while blocking:
                time.sleep(10)
        except KeyboardInterrupt:
            print("Got KeyboardInterrupt, shutting down...")
            serve.shutdown()
            sys.exit()
        except Exception:
            traceback.print_exc()
            print(
                "Received unexpected error, see console logs for more details. Shutting "
                "down..."
            )
            serve.shutdown()
            sys.exit()
