import sys
import time
import traceback
from pathlib import Path

import ray
from ray import serve
from ray.serve.config import HTTPOptions
from ray.serve.deployment import Application, Deployment

from aana.api.api_generation import Endpoint
from aana.api.event_handlers.event_handler import EventHandler
from aana.api.request_handler import RequestHandler
from aana.configs.db import run_alembic_migrations
from aana.configs.settings import settings as aana_settings


class AanaSDK:
    """Aana SDK to deploy and manage Aana deployments and endpoints."""

    def __init__(self, name: str = "app"):
        """Aana SDK to deploy and manage Aana deployments and endpoints.

        Args:
            name (str, optional): The name of the application. Defaults to "app".
        """
        self.name = name
        self.endpoints: dict[str, Endpoint] = {}
        self.deployments: dict[str, Deployment] = {}

        run_alembic_migrations(aana_settings)

    def connect(
        self,
        port: int = 8000,
        host: str = "127.0.0.1",
        address: str = "auto",
        show_logs: bool = False,
        num_cpus: int | None = None,
        num_gpus: int | None = None,
    ):
        """Connect to a Ray cluster or start a new Ray cluster and Ray Serve.

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

        serve_status = serve.status()
        if serve_status.proxies == {}:  # If serve is not running yet
            # TODO: check if the port is already in use if serve is not running yet or
            # check if the port is the same as an existing serve instance if serve is running
            serve.start(http_options=HTTPOptions(port=self.port, host=self.host))

    def show_status(self, app_name: str):
        """Show the status of the application.

        Args:
            app_name (str): The name of the application.
        """

        def print_header(title):
            print(f"\n{'=' * 55}\n{title}\n{'=' * 55}")

        def print_status(deployment, status):
            print(f"{deployment:<30} Status: {status}")

        status = serve.status()
        app_status = status.applications[app_name]

        print_header(f"Application {app_name}")
        print_status(app_name, app_status.status.value)

        for deployment_name, deployment_status in app_status.deployments.items():
            print_header(f"Deployment {deployment_name}")
            print_status("Status", deployment_status.status.value)
            print(f"\n{deployment_status.message}\n")

    def register_deployment(
        self,
        name: str,
        instance: Deployment,
        blocking: bool = False,
        deploy: bool = False,
    ):
        """Register a deployment.

        Args:
            name (str): The name of the deployment.
            instance (Deployment): The instance of the deployment to be registered.
            blocking (bool, optional): If True, the function will block until deployment is complete. Defaults to False.
            deploy (bool, optional): If True, the deployment will be deployed immediately,
                    otherwise it will be registered and can be deployed later when deploy() is called. Defaults to False.
        """
        if deploy:
            try:
                serve.run(
                    instance.bind(),
                    name=name,
                    route_prefix=f"/{name}",
                    blocking=blocking,
                )
            except RuntimeError:
                self.show_status(name)
        else:
            self.deployments[name] = instance

    def get_deployment_app(self, name: str) -> Application:
        """Get the application instance for the deployment.

        Args:
            name (str): The name of the deployment.

        Returns:
            Application: The application instance for the deployment.

        Raises:
            KeyError: If the deployment is not found.
        """
        if name in self.deployments:
            return self.deployments[name].options(route_prefix=f"/{name}").bind()
        else:
            raise KeyError(f"Deployment {name} not found.")  # noqa: TRY003

    def unregister_deployment(self, name: str):
        """Unregister a deployment.

        Args:
            name (str): The name of the deployment to be unregistered.
        """
        if name in self.deployments:
            del self.deployments[name]
        serve.delete(name)

    def get_main_app(self) -> Application:
        """Get the main application instance.

        Returns:
            Application: The main application instance.
        """
        return RequestHandler.options(num_replicas=aana_settings.num_workers).bind(
            endpoints=self.endpoints.values()
        )

    def register_endpoint(
        self,
        name: str,
        path: str,
        summary: str,
        endpoint_cls: type[Endpoint],
        event_handlers: list[EventHandler] | None = None,
    ):
        """Register an endpoint.

        Args:
            name (str): The name of the endpoint.
            path (str): The path of the endpoint.
            summary (str): The summary of the endpoint.
            endpoint_cls (Type[Endpoint]): The class of the endpoint.
            event_handlers (list[EventHandler], optional): The event handlers to register for the endpoint.
        """
        endpoint = endpoint_cls(
            name=name,
            path=path,
            summary=summary,
            event_handlers=event_handlers,
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
        """Deploy the application with the registered endpoints and deployments.

        Args:
            blocking (bool, optional): If True, the function will block until interrupted. Defaults to False.
        """
        for deployment_name in self.deployments:
            try:
                serve.run(
                    self.get_deployment_app(deployment_name),
                    name=deployment_name,
                    route_prefix=f"/{deployment_name}",
                    blocking=False,
                )
            except RuntimeError:  # noqa: PERF203
                self.show_status(deployment_name)

        try:
            serve.run(
                self.get_main_app(),
                name=self.name,
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
        except RuntimeError:
            self.show_status("RequestHandler")
        except Exception:
            traceback.print_exc()
            print(
                "Received unexpected error, see console logs for more details. Shutting "
                "down..."
            )

    def shutdown(self):
        """Shutdown the Aana server."""
        serve.shutdown()
        ray.shutdown()

    def build(self, import_path: str, host: str, port: int, output_dir: Path | str):
        """Build the application configuration file.

        Args:
            import_path (str): The import path of the application.
            host (str): The host to run the application on.
            port (int): The port to run the application on.
            output_dir (Path | str): The output directory to write the config file to.
        """
        import yaml
        from ray._private.utils import import_attr
        from ray.serve._private.deployment_graph_build import build as pipeline_build
        from ray.serve._private.deployment_graph_build import (
            get_and_validate_ingress_deployment,
        )
        from ray.serve.deployment import Application, deployment_to_schema
        from ray.serve.schema import (
            LoggingConfig,
            ServeApplicationSchema,
            ServeDeploySchema,
        )
        from ray.serve.scripts import ServeDeploySchemaDumper

        output_dir = Path(output_dir)

        def build_app_config(import_path: str, name: str):
            app: Application = import_attr(import_path)
            if not isinstance(app, Application):
                raise TypeError(  # noqa: TRY003
                    f"Expected '{import_path}' to be an Application but got {type(app)}."
                )

            deployments = pipeline_build(app, name)
            ingress = get_and_validate_ingress_deployment(deployments)
            schema = ServeApplicationSchema(
                name=name,
                route_prefix=ingress.route_prefix,
                import_path=import_path,
                runtime_env={},
                deployments=[
                    deployment_to_schema(d, include_route_prefix=False)
                    for d in deployments
                ],
            )

            return schema.dict(exclude_unset=True)

        config_str = ""

        app_configs = []
        for deployment_name in self.deployments:
            app_name = deployment_name
            app_configs.append(
                build_app_config(f"{import_path}:{app_name}", name=app_name)
            )

        main_app_name = self.name
        app_configs.append(
            build_app_config(f"{import_path}:{main_app_name}", name=main_app_name)
        )

        deploy_config = {
            "proxy_location": "EveryNode",
            "http_options": {
                "host": host,
                "port": port,
            },
            "logging_config": LoggingConfig().dict(),
            "applications": app_configs,
        }

        # Parse + validate the set of application configs
        ServeDeploySchema.parse_obj(deploy_config)

        config_str += yaml.dump(
            deploy_config,
            Dumper=ServeDeploySchemaDumper,
            default_flow_style=False,
            sort_keys=False,
        )

        # Ensure file ends with only one newline
        config_str = config_str.rstrip("\n") + "\n"

        output_path = output_dir / "config.yaml"
        with output_path.open("w") as f:
            f.write(config_str)
        print(f"Config file written to {output_path}")
