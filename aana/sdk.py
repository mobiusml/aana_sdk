import importlib
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import ray
import yaml
from ray import serve
from ray.autoscaler.v2.schema import ResourceDemand
from ray.autoscaler.v2.sdk import get_cluster_status
from ray.serve.config import HTTPOptions
from ray.serve.deployment import Application, Deployment
from ray.serve.schema import ApplicationStatusOverview
from rich import print as rprint

from aana.api.api_generation import Endpoint
from aana.api.event_handlers.event_handler import EventHandler
from aana.api.request_handler import RequestHandler
from aana.configs.settings import settings as aana_settings
from aana.exceptions.runtime import (
    DeploymentException,
    EmptyMigrationsException,
    FailedDeployment,
    InsufficientResources,
)
from aana.storage.op import run_alembic_migrations
from aana.utils.core import import_from_path


class AanaSDK:
    """Aana SDK to deploy and manage Aana deployments and endpoints."""

    def __init__(self, name: str = "app", migration_func: Callable | None = None):
        """Aana SDK to deploy and manage Aana deployments and endpoints.

        Args:
            name (str, optional): The name of the application. Defaults to "app".
            migration_func (Callable | None): The migration function to run. Defaults to None.
        """
        self.name = name
        self.migration_func = migration_func
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
        if self.migration_func:
            try:
                self.migration_func(aana_settings)
            except EmptyMigrationsException:
                print(
                    "No versions found in the custom migrations. Using default migrations."
                )
                run_alembic_migrations(aana_settings)
        else:
            run_alembic_migrations(aana_settings)

    def print_app_status(self, app_name: str, app_status: ApplicationStatusOverview):
        """Show the status of the application using simple ASCII formatting.

        Args:
            app_name (str): The name of the application.
            app_status (ApplicationStatusOverview): The status of the application.
        """

        def print_separator(end="\n"):
            print("=" * 60, end=end)

        def print_header(title):
            print_separator()
            print(title)
            print_separator()

        def print_key_value(key, value, indent=0):
            print(f"{' ' * indent}{key}: {value}")

        if app_status.deployments:
            for deployment_name, deployment_status in app_status.deployments.items():
                print_header(f"{deployment_name} ({app_name})")
                print_key_value("Status", deployment_status.status.value, indent=0)
                print_key_value("Message", deployment_status.message, indent=0)
        print_separator()

    def add_task_queue(self, deploy: bool = False):
        """Add a task queue deployment.

        Args:
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
        )

    def register_deployment(
        self,
        name: str,
        instance: Deployment,
        deploy: bool = False,
    ):
        """Register a deployment.

        Args:
            name (str): The name of the deployment.
            instance (Deployment): The instance of the deployment to be registered.
            deploy (bool, optional): If True, the deployment will be deployed immediately,
                    otherwise it will be registered and can be deployed later when deploy() is called. Defaults to False.
        """
        if deploy:
            try:
                serve.api._run(
                    instance.bind(),
                    name=name,
                    route_prefix=f"/{name}",
                    _blocking=False,
                )
                self.wait_for_deployment()
            except FailedDeployment:
                status = serve.status()
                app_status = status.applications[name]
                self.print_app_status(name, app_status)
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
            app_name=self.name,
            endpoints=self.endpoints.values(),
            deployments=list(self.deployments.keys()),
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

    def wait_for_deployment(self):  # noqa: C901
        """Wait for the deployment to complete."""
        consecutive_resource_unavailable = 0
        # Number of consecutive checks before raising an resource unavailable error
        resource_unavailable_threshold = 5

        while True:
            status = serve.status()
            if all(
                application.status == "RUNNING"
                for application in status.applications.values()
            ):
                break
            if any(
                application.status == "DEPLOY_FAILED"
                or application.status == "UNHEALTHY"
                for application in status.applications.values()
            ):
                error_messages = []
                for app_name, app_status in status.applications.items():
                    if (
                        app_status.status == "DEPLOY_FAILED"
                        or app_status.status == "UNHEALTHY"
                    ):
                        for (
                            deployment_name,
                            deployment_status,
                        ) in app_status.deployments.items():
                            error_messages.append(
                                f"Error: {deployment_name} ({app_name}): {deployment_status.message}"
                            )
                raise FailedDeployment("\n".join(error_messages))

            gcs_address = ray.get_runtime_context().gcs_address
            cluster_status = get_cluster_status(gcs_address)
            demands = (
                cluster_status.resource_demands.cluster_constraint_demand
                + cluster_status.resource_demands.ray_task_actor_demand
                + cluster_status.resource_demands.placement_group_demand
            )

            resource_unavailable = False
            for demand in demands:
                if isinstance(demand, ResourceDemand) and demand.bundles_by_count:
                    error_message = f"Error: No available node types can fulfill resource request {demand.bundles_by_count[0].bundle}. "
                    if "GPU" in demand.bundles_by_count[0].bundle:
                        error_message += "Might be due to insufficient or misconfigured CPU or GPU resources."
                    resource_unavailable = True
                else:
                    error_message = f"Error: {demand}"
                    resource_unavailable = True

            if resource_unavailable:
                consecutive_resource_unavailable += 1
                if consecutive_resource_unavailable >= resource_unavailable_threshold:
                    raise InsufficientResources(error_message)
            else:
                consecutive_resource_unavailable = 0

            time.sleep(1)  # Wait for 1 second before checking again

    def deploy(self, blocking: bool = False):
        """Deploy the application with the registered endpoints and deployments.

        Args:
            blocking (bool, optional): If True, the function will block until interrupted. Defaults to False.
        """
        try:
            for deployment_name in self.deployments:
                serve.api._run(
                    self.get_deployment_app(deployment_name),
                    name=deployment_name,
                    route_prefix=f"/{deployment_name}",
                    _blocking=False,
                )

            serve.api._run(
                self.get_main_app(),
                name=self.name,
                route_prefix="/",
                _blocking=False,  # blocking manually after to display the message "Deployed successfully."
            )

            self.wait_for_deployment()

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
        except DeploymentException as e:
            status = serve.status()
            serve.shutdown()
            for app_name, app_status in status.applications.items():
                if (
                    app_status.status == "DEPLOY_FAILED"
                    or app_status.status == "UNHEALTHY"
                ):
                    self.print_app_status(app_name, app_status)
            if isinstance(e, InsufficientResources):
                rprint(f"[red] {e} [/red]")
            raise
        except Exception:
            serve.shutdown()
            traceback.print_exc()
            print(
                "Received unexpected error, see console logs for more details. "
                "Shutting down..."
            )
            raise

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
