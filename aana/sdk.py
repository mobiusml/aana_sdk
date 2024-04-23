import sys
import time
import traceback

import ray
from ray import serve
from ray.serve.config import HTTPOptions
from ray.serve.deployment import Deployment

from aana.api.api_generation import Endpoint
from aana.api.request_handler import RequestHandler
from aana.configs.db import run_alembic_migrations
from aana.configs.settings import settings as aana_settings


class AanaSDK:
    """Aana SDK to deploy and manage Aana deployments and endpoints."""

    def __init__(
        self,
        port: int = 8000,
        host: str = "127.0.0.1",
        address: str = "auto",
        show_logs: bool = False,
        num_cpus: int | None = None,
        num_gpus: int | None = None,
    ):
        """Aana SDK to deploy and manage Aana deployments and endpoints.

        Args:
            port (int, optional): The port to run the Aana server on. Defaults to 8000.
            host (str, optional): The host to run the Aana server on. Defaults to "127.0.0.1".
            address (str, optional): The address of the Ray cluster. Defaults to "auto".
            show_logs (bool, optional): If True, the logs will be shown, otherwise
                they will be hidden but can be accessed in the Ray dashboard. Defaults to False.
            num_cpus (int, optional): Number of CPUs the user wishes to assign to each
                raylet. By default, this is set based on virtual cores.
            num_gpus (int, optional): Number of GPUs the user wishes to assign to each
                raylet. By default, this is set based on detected GPUs.
        """
        self.port = port
        self.host = host
        self.endpoints: dict[str, Endpoint] = {}

        run_alembic_migrations(aana_settings)

        try:
            # Try to connect to an existing Ray cluster
            ray.init(
                address=address,
                ignore_reinit_error=True,
                log_to_driver=show_logs,
            )
        except ConnectionError:
            # If connection fails, start a new Ray cluster and serve instance
            ray.init(
                ignore_reinit_error=True,
                log_to_driver=show_logs,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
            )

        # TODO: check if the port is already in use if serve is not running yet or
        # check if the port is the same as an existing serve instance if serve is running
        serve.start(http_options=HTTPOptions(port=self.port, host=self.host))

    def register_deployment(
        self,
        name: str,
        instance: Deployment,
        blocking: bool = False,
    ):
        """Register a deployment.

        Args:
            name (str): The name of the deployment.
            instance (Deployment): The instance of the deployment to be registered.
            blocking (bool, optional): If True, the function will block until deployment is complete. Defaults to False.
        """
        serve.run(
            instance.bind(),
            name=name,
            route_prefix=f"/{name}",
            blocking=blocking,
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
                name="RequestHandler",
                route_prefix="/",
                blocking=False,  # blocking manually after to display the message "Deployed successfully."
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

    def shutdown(self):
        """Shutdown the Aana server."""
        serve.shutdown()
        ray.shutdown()
