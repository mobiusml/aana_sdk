import json
import pprint
from typing import Any

from jinja2 import Environment, PackageLoader


def add_custom_schemas_to_openapi_schema(
    openapi_schema: dict[str, Any], custom_schemas: dict[str, Any]
) -> dict[str, Any]:
    """Add custom schemas to the openapi schema.

    File upload is that FastAPI doesn't support Pydantic models in multipart requests.
    There is a discussion about it on FastAPI discussion forum.
    See https://github.com/tiangolo/fastapi/discussions/8406
    The topic starter suggests a workaround.
    The workaround is to use Forms instead of Pydantic models in the endpoint definition and
    then convert the Forms to Pydantic models in the endpoint itself
    using parse_raw_as function from Pydantic.
    Since Pydantic model isn't used in the endpoint definition,
    the API documentation will not be generated automatically.
    So the workaround also suggests updating the API documentation manually
    by overriding the openapi method of a FastAPI application.

    Args:
        openapi_schema (dict): The openapi schema.
        custom_schemas (dict): The custom schemas.

    Returns:
        dict: The openapi schema with the custom schemas added.
    """
    if "$defs" not in openapi_schema:
        openapi_schema["$defs"] = {}
    for schema_name, schema in custom_schemas.items():
        # if we have a definitions then we need to move them out to the top level of the schema
        if "$defs" in schema:
            openapi_schema["$defs"].update(schema["$defs"])
            del schema["$defs"]
        openapi_schema["components"]["schemas"][f"Body_{schema_name}"]["properties"][
            "body"
        ] = schema
    return openapi_schema


def rewrite_anyof(obj: dict | list) -> None:
    """Recursively traverse the object (dict or list) and rewrite anyOf patterns.

    If an object has an "anyOf" key with exactly one non-null schema and one or more null types,
    it replaces it with that schema plus 'nullable': True.
    """
    if isinstance(obj, dict):
        # Check if the current dict has an 'anyOf' property
        if "anyOf" in obj and isinstance(obj["anyOf"], list):
            anyof = obj["anyOf"]
            # Separate schemas that are not null and that are null
            non_null = [schema for schema in anyof if schema.get("type") != "null"]
            null_schemas = [schema for schema in anyof if schema.get("type") == "null"]
            # Only rewrite if exactly one non-null type is found along with at least one null.
            if len(non_null) == 1 and null_schemas:
                new_schema = dict(non_null[0])  # Copy the non-null schema
                new_schema["nullable"] = True  # Mark it as nullable
                # Delete the 'anyOf' key and replace it with the new schema
                obj.pop("anyOf")
                obj.update(new_schema)
        # Recursively check every value in the dict.
        for _key, value in list(obj.items()):
            rewrite_anyof(value)
    elif isinstance(obj, list):
        for item in obj:
            rewrite_anyof(item)


def resolve_ref(schema: dict, root: dict) -> dict:
    """Resolves a JSON Schema $ref from the root schema."""
    ref = schema.get("$ref")
    if ref and ref.startswith("#/"):
        parts = ref.lstrip("#/").split("/")
        resolved = root
        for part in parts:
            resolved = resolved.get(part)
            if resolved is None:
                break
        return resolved
    return schema


def generate_example(schema: dict, root_schema: dict):  # noqa: C901
    """Recursively generates an example from a JSON Schema."""
    if root_schema is None:
        root_schema = schema

    # Store the default value before potentially resolving a reference
    default_value = schema.get("default")

    # Resolve references
    if "$ref" in schema:
        resolved_schema = resolve_ref(schema, root_schema)
        if resolved_schema is not None:
            # If the original schema had a default value, prioritize it
            if default_value is not None:
                return default_value
            # Check if this is an enum
            if "enum" in resolved_schema and resolved_schema["enum"]:
                # First check if resolved schema has a default
                if "default" in resolved_schema:
                    return resolved_schema["default"]
                # Otherwise use the first enum value
                return resolved_schema["enum"][0]
            return generate_example(resolved_schema, root_schema)
        return None

    # Use the provided examples, example, or default if available
    if (
        "examples" in schema
        and isinstance(schema["examples"], list)
        and schema["examples"]
    ):
        return schema["examples"][0]
    if "example" in schema:
        return schema["example"]
    if "default" in schema:
        return schema["default"]

    # Handle enum types explicitly
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]  # Return the first enum value as default

    # Handle alternative schemas (anyOf/oneOf)
    for key in ("anyOf", "oneOf"):
        if key in schema:
            for subschema in schema[key]:
                example = generate_example(subschema, root_schema)
                if example is not None:
                    return example
            return None

    # Fallback based on type
    schema_type = schema.get("type")
    if schema_type == "object":
        example = {}
        properties = schema.get("properties", {})
        for prop, prop_schema in properties.items():
            example[prop] = generate_example(prop_schema, root_schema)
        return example
    elif schema_type == "array":
        items_schema = schema.get("items", {})
        return [generate_example(items_schema, root_schema)]
    elif schema_type == "string":
        if schema.get("format") == "uri":
            return "https://example.com"
        return "string"
    elif schema_type == "integer":
        return 0
    elif schema_type == "number":
        return 0.0
    elif schema_type == "boolean":
        return True

    return None


def add_code_samples_to_endpoints(openapi_spec: dict) -> dict:
    """Add code samples to endpoints.

    Iterates over all endpoints in the OpenAPI spec and adds code samples (cURL and Python)
    for endpoints that accept 'application/x-www-form-urlencoded' request bodies.

    Args:
      openapi_spec (dict): The full OpenAPI specification.

    Returns:
      dict: The modified OpenAPI specification with code samples added.
    """
    # Set up the Jinja2 environment to load your code sample template.
    loader = PackageLoader("aana.utils", "openapi_code_templates")
    env = Environment(loader=loader, autoescape=False)  # noqa: S701
    curl_template = env.get_template("curl_form.j2")
    non_streaming_python_template = env.get_template("python_form.j2")
    streaming_python_template = env.get_template("python_form_streaming.j2")

    # Extract the base URL from servers if available
    base_url = "http://localhost:8000"
    if openapi_spec.get("servers"):
        base_url = openapi_spec["servers"][0].get("url", base_url)

    # Iterate over all paths and methods in the spec.
    for path, path_item in openapi_spec.get("paths", {}).items():
        for method, operation in path_item.items():
            # Only process HTTP methods.
            if method.lower() not in [
                "get",
                "post",
                "put",
                "delete",
                "patch",
                "options",
                "head",
                "trace",
            ]:
                continue

            # Check if the endpoint has a requestBody.
            request_body = operation.get("requestBody", {})
            content = request_body.get("content", {})
            if "application/x-www-form-urlencoded" not in content:
                continue

            # Skip if the code sample is already provided.
            if "x-codeSamples" in operation:
                continue

            response = operation.get("responses", {}).get("200", {})
            response_content_type = response.get("content", {}).keys()
            is_streaming = "application/x-ndjson" in response_content_type

            # Extract the JSON schema for the form data.
            schema = content["application/x-www-form-urlencoded"].get("schema", {})

            # Generate an example payload using your generate_example() function.
            example = generate_example(schema, openapi_spec)

            # Remove body key from the example payload.
            example = example.get("body", example)

            # Render the code snippets using the provided templates
            curl_snippet = curl_template.render(
                base_url=base_url,
                path=path,
                body=json.dumps(example, separators=(",", ":")),
            )

            if is_streaming:
                python_template = streaming_python_template
            else:
                python_template = non_streaming_python_template
            python_snippet = python_template.render(
                base_url=base_url, path=path, body=pprint.pformat(example, width=40)
            )

            # Attach the code samples to the endpoint.
            operation["x-codeSamples"] = [
                {"lang": "cURL", "source": curl_snippet, "label": "cURL"},
                {"lang": "Python", "source": python_snippet, "label": "Python"},
            ]

    return openapi_spec
