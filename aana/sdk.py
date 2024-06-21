import importlib
import sys
import time
import traceback
from pathlib import Path

import ray
import yaml
from ray import serve
from ray.serve.config import HTTPOptions
from ray.serve.deployment import Application, Deployment
from rich import print as rprint

from aana.api.api_generation import Endpoint
from aana.api.event_handlers.event_handler import EventHandler
from aana.api.request_handler import RequestHandler
from aana.configs.settings import settings as aana_settings
from aana.storage.op import run_alembic_migrations
from aana.utils.core import import_from_path


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

        if aana_settings.task_queue.enabled:
            self.add_task_queue(deploy=False)

    def connect(
        self,
        port: int = 8000,
        host: str = "127.0.0.1",
        address: str = "auto",
        show_logs: bool = False,
        num_cpus: int | None = None,
        num_gpus: int | None = None,
    ) -> "AanaSDK":
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

        return self

    def migrate(self):
        """Run Alembic migrations."""
        run_alembic_migrations(aana_settings)

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

    def add_task_queue(self, blocking: bool = False, deploy: bool = False):
        """Add a task queue deployment.

        Args:
            blocking (bool, optional): If True, the function will block until deployment is complete. Defaults to False.
            deploy (bool, optional): If True, the deployment will be deployed immediately,
                    otherwise it will be registered and can be deployed later when deploy() is called. Defaults to False.
        """
        from aana.deployments.task_queue_deployment import (
            TaskQueueConfig,
            TaskQueueDeployment,
        )

        task_queue_deployment = TaskQueueDeployment.options(
            num_replicas=1,
            user_config=TaskQueueConfig(
                app_name=self.name,
            ).model_dump(mode="json"),
        )
        self.register_deployment(
            "task_queue_deployment",
            task_queue_deployment,
            deploy=deploy,
            blocking=blocking,
        )

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
            rprint("[green]Deployed successfully.[/green]")
            rprint(
                f"Documentation is available at "
                f"[link=http://{self.host}:{self.port}/docs]http://{self.host}:{self.port}/docs[/link] and "
                f"[link=http://{self.host}:{self.port}/redoc]http://{self.host}:{self.port}/redoc[/link]"
            )
            while blocking:
                time.sleep(10)
        except KeyboardInterrupt:
            print("Got KeyboardInterrupt, shutting down...")
            serve.shutdown()
            sys.exit()
        except RuntimeError:
            self.show_status(self.name)
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

    def build(
        self,
        import_path: str,
        host: str = "0.0.0.0",  # noqa: S104
        port: int = 8000,
        app_config_name: str = "app_config",
        config_name: str = "config",
    ):
        """Build the application configuration file.

        Two files will be created: app_config (.py) and config (.yaml).s

        Args:
            import_path (str): The import path of the application.
            host (str): The host to run the application on. Defaults to "0.0.0.0".
            port (int): The port to run the application on. Defaults to 8000.
            app_config_name (str): The name of the application config file. Defaults to "app_config".
            config_name (str): The name of the config file. Defaults to "config".
        """
        # Split the import path into module and variable.
        # For example, aana.projects.whisper.app:aana_app will be split into
        # module "aana.projects.whisper.app" and variable "aana_app".
        app_module, app_var = import_path.split(":")

        # Use location of the app module as the output directory
        output_dir = Path(importlib.util.find_spec(app_module).origin).parent

        # Import AanaSDK app from the given import path
        aana_app = import_from_path(import_path)
        if not isinstance(aana_app, AanaSDK):
            raise TypeError(  # noqa: TRY003
                f"Error: {import_path} is not an AanaSDK instance, got {type(aana_app)}"
            )

        # Generate the app config file
        # Example:
        #    from aana.projects.whisper.app import aana_app
        #    asr_deployment = aana_app.get_deployment_app("asr_deployment")
        #    vad_deployment = aana_app.get_deployment_app("vad_deployment")
        #    whisper_app = aana_app.get_main_app()
        app_config = ""
        app_config += f"from {app_module} import {app_var}\n\n"
        for deployment_name in aana_app.deployments:
            app_config += f"{deployment_name} = {app_var}.get_deployment_app('{deployment_name}')\n"
        app_config += f"{aana_app.name} = {app_var}.get_main_app()\n"

        # Output path for the app config file
        app_config_path = output_dir / f"{app_config_name}.py"
        # Import path for the app config file, for example aana.projects.whisper.app_config
        app_config_import_path = f"{app_module.rsplit('.', 1)[0]}.{app_config_name}"

        # Write the app config file
        with app_config_path.open("w") as f:
            f.write(app_config)

        # Build "serve build" command to generate config.yaml
        # For example,
        # serve build aana.projects.whisper.app_config:vad_deployment
        #             aana.projects.whisper.app_config:asr_deployment
        #             aana.projects.whisper.app_config:whisper_app
        #             -o /workspaces/aana_sdk/aana/projects/whisper/config.yaml
        config_path = output_dir / f"{config_name}.yaml"
        serve_options = []
        for deployment_name in aana_app.deployments:
            serve_options.append(f"{app_config_import_path}:{deployment_name}")  # noqa: PERF401
        serve_options += [
            f"{app_config_import_path}:{aana_app.name}",
            "--output-path",
            str(config_path),
            "--app-dir",
            output_dir,
        ]

        # Execute "serve build" with click CliRuuner
        from click.testing import CliRunner
        from ray.serve.scripts import ServeDeploySchemaDumper, build

        result = CliRunner().invoke(build, serve_options)
        if result.exception:
            raise result.exception

        # Update the config file with the host and port and rename apps
        with config_path.open() as f:
            config = yaml.load(f, Loader=yaml.FullLoader)  # noqa: S506

        config["http_options"] = {"host": host, "port": port}

        for app in config["applications"]:
            app["name"] = app["import_path"].split(":")[-1]

        with config_path.open("w") as f:
            yaml.dump(
                config,
                f,
                Dumper=ServeDeploySchemaDumper,
                default_flow_style=False,
                sort_keys=False,
            )

        print(f"App config successfully saved to {app_config_path}")
        print(f"Config successfully saved to {config_path}")
