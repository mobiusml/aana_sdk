from typing import Any

from fastapi.openapi.utils import get_openapi
from ray import serve

from aana.api.api_generation import Endpoint, add_custom_schemas_to_openapi_schema
from aana.api.app import app
from aana.api.event_handlers.event_manager import EventManager
from aana.api.responses import AanaJSONResponse
from aana.utils.typing import is_async_generator


@serve.deployment(ray_actor_options={"num_cpus": 0.1})
@serve.ingress(app)
class RequestHandler:
    """This class is used to handle requests to the Aana application."""

    ready = False

    def __init__(self, endpoints: list[Endpoint]):
        """Constructor.

        Args:
            endpoints (dict): List of endpoints for the request
        """
        self.endpoints = endpoints

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

    async def call_endpoint(self, path: str, **kwargs: dict[str, Any]) -> Any:
        """Call the endpoint from FastAPI with the given name.

        Args:
            path (str): The path of the endpoint.
            **kwargs: The arguments to pass to the endpoint.

        Returns:
            Any: The response from the endpoint.
        """
        endpoint = None
        for e in self.endpoints:
            if e.path == path:
                endpoint = e
                break
        if endpoint is None:
            raise ValueError(f"Endpoint {path} not found")  # noqa: TRY003

        if not endpoint.initialized:
            await endpoint.initialize()

        annotations = endpoint.run.__annotations__
        return_type = annotations.get("return", None)

        if is_async_generator(return_type):
            return [item async for item in endpoint.run(**kwargs)]
        else:
            return await endpoint.run(**kwargs)
