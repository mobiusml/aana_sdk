import json
import time
from typing import Annotated, Any
from uuid import uuid4

import orjson
import ray
from fastapi import Depends
from fastapi.openapi.utils import get_openapi
from fastapi.responses import StreamingResponse
from ray import serve
from sqlalchemy.orm import Session

from aana.api.api_generation import Endpoint, add_custom_schemas_to_openapi_schema
from aana.api.app import app
from aana.api.event_handlers.event_manager import EventManager
from aana.api.exception_handler import custom_exception_handler
from aana.api.responses import AanaJSONResponse
from aana.configs.settings import settings as aana_settings
from aana.core.models.api import DeploymentStatus, SDKStatus, SDKStatusResponse
from aana.core.models.chat import ChatCompletion, ChatCompletionRequest, ChatDialog
from aana.core.models.sampling import SamplingParams
from aana.core.models.task import TaskId, TaskInfo
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.storage.models.task import Status as TaskStatus
from aana.storage.repository.task import TaskRepository
from aana.storage.session import get_session


def get_db():
    """Get a database session."""
    db = get_session()
    try:
        yield db
    finally:
        db.close()


@serve.deployment(ray_actor_options={"num_cpus": 0.1})
@serve.ingress(app)
class RequestHandler:
    """This class is used to handle requests to the Aana application."""

    ready = False

    def __init__(
        self, app_name: str, endpoints: list[Endpoint], deployments: list[str]
    ):
        """Constructor.

        Args:
            app_name (str): The name of the application.
            endpoints (dict): List of endpoints for the request.
            deployments (list[str]): List of deployment names for the app.
        """
        self.app_name = app_name
        self.endpoints = endpoints
        self.deployments = deployments

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

    def custom_openapi(self) -> dict[str, Any]:
        """Returns OpenAPI schema, generating it if necessary."""
        if app.openapi_schema:
            return app.openapi_schema
        # TODO: populate title and version from package info
        openapi_schema = get_openapi(title="Aana", version="0.1.0", routes=app.routes)
        openapi_schema = add_custom_schemas_to_openapi_schema(
            openapi_schema=openapi_schema, custom_schemas=self.custom_schemas
        )
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    @app.get("/api/ready")
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

    async def execute_task(self, task_id: str) -> Any:
        """Execute a task.

        Args:
            task_id (str): The task ID.

        Returns:
            Any: The response from the endpoint.
        """
        session = get_session()
        task_repo = TaskRepository(session)
        try:
            task = task_repo.read(task_id)
            path = task.endpoint
            kwargs = task.data

            task_repo.update_status(task_id, TaskStatus.RUNNING, 0)

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

            task_repo.update_status(task_id, TaskStatus.COMPLETED, 100, out)
        except Exception as e:
            error_response = custom_exception_handler(None, e)
            error = orjson.loads(error_response.body)
            task_repo.update_status(task_id, TaskStatus.FAILED, 0, error)
        else:
            return out

    @app.get(
        "/tasks/get/{task_id}",
        summary="Get Task Status",
        description="Get the task status by task ID.",
        include_in_schema=aana_settings.task_queue.enabled,
    )
    async def get_task_endpoint(
        self, task_id: str, db: Annotated[Session, Depends(get_db)]
    ) -> TaskInfo:
        """Get the task with the given ID.

        Args:
            task_id (str): The ID of the task.
            db (Session): The database session.

        Returns:
            TaskInfo: The status of the task.
        """
        task_repo = TaskRepository(db)
        task = task_repo.read(task_id)
        return TaskInfo(
            id=str(task.id),
            status=task.status,
            result=task.result,
        )

    @app.get(
        "/tasks/delete/{task_id}",
        summary="Delete Task",
        description="Delete the task by task ID.",
        include_in_schema=aana_settings.task_queue.enabled,
    )
    async def delete_task_endpoint(
        self, task_id: str, db: Annotated[Session, Depends(get_db)]
    ) -> TaskId:
        """Delete the task with the given ID.

        Args:
            task_id (str): The ID of the task.
            db (Session): The database session.

        Returns:
            TaskInfo: The deleted task.
        """
        task_repo = TaskRepository(db)
        task = task_repo.delete(task_id)
        return TaskId(task_id=str(task.id))

    @app.post("/chat/completions", response_model=ChatCompletion)
    async def chat_completions(self, request: ChatCompletionRequest):
        """Handle chat completions requests for OpenAI compatible API."""

        async def _async_chat_completions(
            handle: AanaDeploymentHandle,
            dialog: ChatDialog,
            sampling_params: SamplingParams,
        ):
            async for response in handle.chat_stream(
                dialog=dialog, sampling_params=sampling_params
            ):
                chunk = {
                    "id": f"chatcmpl-{uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "model": request.model,
                    "created": int(time.time()),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": response["text"], "role": "assistant"},
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        # Check if the deployment exists
        try:
            handle = await AanaDeploymentHandle.create(request.model)
        except ray.serve.exceptions.RayServeException:
            return AanaJSONResponse(
                content={
                    "error": {"message": f"The model `{request.model}` does not exist."}
                },
                status_code=404,
            )

        # Check if the deployment is a chat model
        if not hasattr(handle, "chat") or not hasattr(handle, "chat_stream"):
            return AanaJSONResponse(
                content={
                    "error": {"message": f"The model `{request.model}` does not exist."}
                },
                status_code=404,
            )

        dialog = ChatDialog(
            messages=request.messages,
        )

        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
        )

        if request.stream:
            return StreamingResponse(
                _async_chat_completions(handle, dialog, sampling_params),
                media_type="application/x-ndjson",
            )
        else:
            response = await handle.chat(dialog=dialog, sampling_params=sampling_params)
            return {
                "id": f"chatcmpl-{uuid4().hex}",
                "object": "chat.completion",
                "model": request.model,
                "created": int(time.time()),
                "choices": [{"index": 0, "message": response["message"]}],
            }

    @app.get("/api/status", response_model=SDKStatusResponse)
    async def status(self) -> SDKStatusResponse:
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
