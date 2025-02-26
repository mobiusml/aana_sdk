from typing import Any
from uuid import UUID

import orjson
from fastapi import APIRouter
from fastapi.openapi.utils import get_openapi
from ray import serve

# from scalar_fastapi import get_scalar_api_reference
from aana.api.api_generation import Endpoint
from aana.api.app import app
from aana.api.event_handlers.event_manager import EventManager
from aana.api.exception_handler import custom_exception_handler
from aana.api.responses import AanaJSONResponse
from aana.api.security import AdminAccessDependency
from aana.core.models.api import DeploymentStatus, SDKStatus, SDKStatusResponse
from aana.routers.openai import router as openai_router
from aana.routers.task import router as task_router
from aana.routers.webhook import (
    WebhookEventType,
    trigger_task_webhooks,
)
from aana.routers.webhook import router as webhook_router
from aana.storage.models.task import Status as TaskStatus
from aana.storage.repository.task import TaskRepository
from aana.storage.session import get_session
from aana.utils.openapi import (
    add_code_samples_to_endpoints,
    add_custom_schemas_to_openapi_schema,
    rewrite_anyof,
)


@serve.deployment(ray_actor_options={"num_cpus": 0.1})
@serve.ingress(app)
class RequestHandler:
    """This class is used to handle requests to the Aana application."""

    ready = False

    def __init__(
        self,
        app_name: str,
        endpoints: list[Endpoint],
        deployments: list[str],
        routers: list[APIRouter] | None = None,
        openapi_params: dict[str, Any] | None = None,
    ):
        """Constructor.

        Args:
            app_name (str): The name of the application.
            endpoints (dict): List of endpoints for the request.
            deployments (list[str]): List of deployment names for the app.
            routers (list[APIRouter]): List of FastAPI routers to include in the app.
            openapi_params (dict[str, Any]): Parameters for the OpenAPI schema.
        """
        self.app_name = app_name
        self.endpoints = endpoints
        self.deployments = deployments
        self.openapi_params = openapi_params

        # Include the default routers
        app.include_router(webhook_router)  # For webhook management
        app.include_router(task_router)  # For task management
        app.include_router(openai_router)  # For OpenAI-compatible API

        # Include the custom routers (from Aana Apps)
        if routers is not None:
            for router in routers:
                app.include_router(router)

        self.event_manager = EventManager()
        self.custom_schemas: dict[str, dict] = {}
        for endpoint in self.endpoints:
            endpoint.register(
                app=app,
                custom_schemas=self.custom_schemas,
                event_manager=self.event_manager,
            )

        app.openapi = self.custom_openapi
        self.ready = True
        self.running_tasks = set()

    # @app.get("/scalar", include_in_schema=False)
    # async def scalar_html(self):
    #     """The endpoint for the Scalar API documentation."""
    #     return get_scalar_api_reference(
    #         openapi_url=app.openapi_url, title=app.title, hide_models=True
    #     )

    def custom_openapi(self) -> dict[str, Any]:
        """Returns OpenAPI schema, generating it if necessary."""
        if app.openapi_schema:
            return app.openapi_schema

        if self.openapi_params is None:
            self.openapi_params = {}
        if "title" not in self.openapi_params:
            self.openapi_params["title"] = self.app_name
        if "version" not in self.openapi_params:
            self.openapi_params["version"] = "0.1.0"
        openapi_schema = get_openapi(routes=app.routes, **self.openapi_params)

        # Add the security scheme for x-api-key
        openapi_schema["components"]["securitySchemes"] = {
            "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "x-api-key"}
        }
        openapi_schema["security"] = [{"ApiKeyAuth": []}]

        # Add custom schemas to the openapi schema
        openapi_schema = add_custom_schemas_to_openapi_schema(
            openapi_schema=openapi_schema, custom_schemas=self.custom_schemas
        )

        # # dump the schema to a file for debugging
        # import pickle

        # print("Dumping openapi schema to /workspaces/aana_sdk/openapi.pkl")
        # with open("/workspaces/aana_sdk/openapi.pkl", "wb") as f:
        #     pickle.dump(openapi_schema, f)

        openapi_schema = add_code_samples_to_endpoints(openapi_schema)

        # Rewrite anyOf patterns to include 'nullable': True
        rewrite_anyof(openapi_schema)

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    async def execute_task(self, task_id: str | UUID) -> Any:
        """Execute a task.

        Args:
            task_id (str | UUID): The ID of the task.

        Returns:
            Any: The response from the endpoint.
        """
        try:
            self.running_tasks.add(task_id)
            with get_session() as session:
                task_repo = TaskRepository(session)
                task = task_repo.read(task_id)
                path = task.endpoint
                kwargs = task.data

                task = task_repo.update_status(task_id, TaskStatus.RUNNING, 0)
                await trigger_task_webhooks(WebhookEventType.TASK_STARTED, task)

            for e in self.endpoints:
                if e.path == path:
                    endpoint = e
                    break
            else:
                raise ValueError(f"Endpoint {path} not found")  # noqa: TRY003, TRY301

            if not endpoint.initialized:
                await endpoint.initialize()

            if endpoint.is_streaming_response():
                out = [item async for item in endpoint.run(**kwargs)]
            else:
                out = await endpoint.run(**kwargs)

            with get_session() as session:
                task = TaskRepository(session).update_status(
                    task_id, TaskStatus.COMPLETED, 100, out
                )
                await trigger_task_webhooks(WebhookEventType.TASK_COMPLETED, task)
        except Exception as e:
            error_response = custom_exception_handler(None, e)
            error = orjson.loads(error_response.body)
            with get_session() as session:
                task = TaskRepository(session).update_status(
                    task_id, TaskStatus.FAILED, 0, error
                )
                await trigger_task_webhooks(WebhookEventType.TASK_FAILED, task)
        else:
            return out
        finally:
            self.running_tasks.remove(task_id)

    @app.get("/api/ready", tags=["system"])
    async def is_ready(self):
        """The endpoint for checking if the application is ready.

        Real reason for this endpoint is to make automatic endpoint generation work.
        If RequestHandler doesn't have any endpoints defined manually,
        then the automatic endpoint generation doesn't work.
        #TODO: Find a better solution for this.

        Returns:
            AanaJSONResponse: The response containing the ready status.
        """
        return AanaJSONResponse(content={"ready": self.ready})

    async def check_health(self):
        """Check the health of the application."""
        # Heartbeat for the running tasks
        with get_session() as session:
            task_repo = TaskRepository(session)
            task_repo.heartbeat(self.running_tasks)

    @app.get("/api/status", response_model=SDKStatusResponse, tags=["system"])
    async def status(self, is_admin: AdminAccessDependency) -> SDKStatusResponse:
        """The endpoint for checking the status of the application."""
        app_names = [
            self.app_name,
            *self.deployments,
        ]  # the list of Ray Serve apps that belong to this Aana app
        serve_status = serve.status()
        app_statuses = {
            app_name: app_status
            for app_name, app_status in serve_status.applications.items()
            if app_name in app_names
        }

        app_status_message = ""
        if any(
            app.status in ["DEPLOY_FAILED", "UNHEALTHY", "NOT_STARTED"]
            for app in app_statuses.values()
        ):
            sdk_status = SDKStatus.UNHEALTHY
            error_messages = []
            for app_name, app_status in app_statuses.items():
                if app_status.status in ["DEPLOY_FAILED", "UNHEALTHY"]:
                    for (
                        deployment_name,
                        deployment_status,
                    ) in app_status.deployments.items():
                        error_messages.append(
                            f"Error: {deployment_name} ({app_name}): {deployment_status.message}"
                        )
            app_status_message = "\n".join(error_messages)
        elif all(app.status == "RUNNING" for app in app_statuses.values()):
            sdk_status = SDKStatus.RUNNING
        elif any(
            app.status in ["DEPLOYING", "DELETING"] for app in app_statuses.values()
        ):
            sdk_status = SDKStatus.DEPLOYING
        else:
            sdk_status = SDKStatus.UNHEALTHY
            app_status_message = "Unknown status"

        deployment_statuses = {}
        for app_name, app_status in app_statuses.items():
            messages = []
            for deployment_name, deployment_status in app_status.deployments.items():
                if deployment_status.message:
                    messages.append(
                        f"{deployment_name} ({app_name}): {deployment_status.message}"
                    )
            message = "\n".join(messages)

            deployment_statuses[app_name] = DeploymentStatus(
                status=app_status.status, message=message
            )

        return SDKStatusResponse(
            status=sdk_status,
            message=app_status_message,
            deployments=deployment_statuses,
        )
