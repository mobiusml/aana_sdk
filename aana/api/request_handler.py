from typing import Dict
from ray import serve
from fastapi.openapi.utils import get_openapi

from mobius_pipeline.pipeline import Pipeline
from aana.api.api_generation import (
    add_custom_schemas_to_openapi_schema,
    get_request_schema,
    register_endpoint,
)

from aana.api.app import app
from aana.api.responses import AanaJSONResponse
from aana.configs.endpoints import endpoints
from aana.configs.pipeline import nodes


@serve.deployment(route_prefix="/", num_replicas=1, ray_actor_options={"num_cpus": 0.1})
@serve.ingress(app)
class RequestHandler:
    """This class is used to handle requests to the Aana application."""

    ready = False

    def __init__(self, deployments: Dict):
        """
        Args:
            deployments (Dict): The dictionary of deployments.
                It is passed to the context to the pipeline so the pipeline can access the deployments handles.
        """
        self.context = {
            "deployments": deployments,
        }
        self.pipeline = Pipeline(nodes, self.context)

        self.custom_schemas = {}
        for endpoint in endpoints:
            register_endpoint(app=app, pipeline=self.pipeline, endpoint=endpoint)
            # get schema for endpoint to add to openapi schema
            schema = get_request_schema(pipeline=self.pipeline, endpoint=endpoint)
            self.custom_schemas[endpoint.name] = schema

        app.openapi = self.custom_openapi
        self.ready = True

    def custom_openapi(self):
        if app.openapi_schema:
            return app.openapi_schema
        # TODO: populate title and version from package info
        openapi_schema = get_openapi(title="Aana", version="0.1.0", routes=app.routes)
        openapi_schema = add_custom_schemas_to_openapi_schema(
            openapi_schema=openapi_schema, custom_schemas=self.custom_schemas
        )
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    def get_context(self):
        """
        Returns:
            dict: The context of the pipeline.
        """
        return self.context

    @app.get("/api/ready")
    async def ready(self):
        """
        The endpoint for checking if the application is ready.

        Real reason for this endpoint is to make automatic endpoint generation work.
        If RequestHandler doesn't have any endpoints defined manually,
        then the automatic endpoint generation doesn't work.
        #TODO: Find a better solution for this.

        Returns:
            AanaJSONResponse: The response containing the ready status.
        """

        return AanaJSONResponse(content={"ready": self.ready})
