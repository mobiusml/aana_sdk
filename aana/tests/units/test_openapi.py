# ruff: noqa: S101

from aana.utils.openapi import generate_example


def test_direct_example():
    """When the schema has a direct 'example', it should be returned."""
    schema = {"type": "string", "example": "test string"}
    assert generate_example(schema, None) == "test string"


def test_default_value():
    """If no 'example' is provided, but a 'default' exists, it should be returned."""
    schema = {"type": "string", "default": "default string"}
    assert generate_example(schema, None) == "default string"


def test_examples_priority():
    """If 'examples' (a list) is provided, the first one should be chosen over 'example' or 'default'."""
    schema = {"type": "integer", "examples": [100, 200], "example": 50, "default": 0}
    assert generate_example(schema, None) == 100


def test_anyof_example():
    """For a schema with 'anyOf', the function should choose the first valid example."""
    schema = {
        "anyOf": [
            {"type": "string", "example": "first alternative"},
            {"type": "string", "example": "second alternative"},
        ]
    }
    assert generate_example(schema, None) == "first alternative"


def test_oneof_example():
    """For a schema with 'oneOf', the function should choose the first valid example."""
    schema = {
        "oneOf": [
            {"type": "integer", "example": 42},
            {"type": "integer", "example": 100},
        ]
    }
    assert generate_example(schema, None) == 42


def test_object_example():
    """For an object type, it should recursively generate examples for each property."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "example": "John Doe"},
            "age": {"type": "integer", "default": 30},
        },
    }
    expected = {"name": "John Doe", "age": 30}
    assert generate_example(schema, None) == expected


def test_array_example():
    """For an array type, it should generate an example list with one example item."""
    schema = {"type": "array", "items": {"type": "string", "example": "item1"}}
    expected = ["item1"]
    assert generate_example(schema, None) == expected


def test_boolean_example():
    """For a boolean type, it should return True as a fallback."""
    schema = {"type": "boolean"}
    assert generate_example(schema, None) is True


def test_number_example():
    """For a number type, it should return 0.0 as a fallback."""
    schema = {"type": "number"}
    assert generate_example(schema, None) == 0.0


def test_uri_format_example():
    """For a string with format 'uri', it should return a sample URL."""
    schema = {"type": "string", "format": "uri"}
    assert generate_example(schema, None) == "https://example.com"


def test_ref_resolution():
    """Test that $ref is resolved correctly from the root schema."""
    root_schema = {
        "$defs": {"MyString": {"type": "string", "example": "ref example"}},
        "type": "object",
        "properties": {"field": {"$ref": "#/$defs/MyString"}},
    }
    expected = {"field": "ref example"}
    assert generate_example(root_schema, root_schema) == expected


def test_complex_image_input_schema():
    """Test a complex schema similar to your ImageInput example."""
    json_schema = {
        "$defs": {
            "ImageInput": {
                "description": (
                    "An image. \nExactly one of 'path', 'url', 'content' or 'numpy' must be provided. \n"
                    "If 'path' is provided, the image will be loaded from the path. \n"
                    "If 'url' is provided, the image will be downloaded from the url. \n"
                    "The 'content' and 'numpy' will be loaded automatically if files are uploaded to the endpoint "
                    "and the corresponding field is set to the file name."
                ),
                "examples": [
                    {"path": "/path/to/image.jpg"},
                    {"url": "https://example.com/image.jpg"},
                ],
                "properties": {
                    "path": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "description": "The file path of the image.",
                        "title": "Path",
                    },
                    "url": {
                        "anyOf": [
                            {"format": "uri", "minLength": 1, "type": "string"},
                            {"type": "null"},
                        ],
                        "default": None,
                        "description": "The URL of the image.",
                        "title": "Url",
                    },
                    "content": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "description": (
                            "The name of the file uploaded to the endpoint. "
                            "The image will be loaded from the file automatically."
                        ),
                        "title": "Content",
                    },
                    "numpy": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "description": (
                            "The name of the file uploaded to the endpoint. "
                            "The image will be loaded from the file automatically."
                        ),
                        "title": "Numpy",
                    },
                    "media_id": {
                        "description": "The ID of the image. If not provided, it will be generated automatically.",
                        "example": "123e4567-e89b-12d3-a456-426614174000",
                        "maxLength": 36,
                        "title": "Media Id",
                        "type": "string",
                    },
                },
                "title": "ImageInput",
                "type": "object",
            }
        },
        "properties": {
            "image": {"$ref": "#/$defs/ImageInput"},
            "media_id": {
                "description": "The media ID.",
                "example": "123e4567-e89b-12d3-a456-426614174000",
                "maxLength": 36,
                "title": "Media Id",
                "type": "string",
            },
        },
        "required": ["image", "media_id"],
        "title": "ImageSizeEndpointRequest",
        "type": "object",
    }
    expected = {
        "image": {"path": "/path/to/image.jpg"},
        "media_id": "123e4567-e89b-12d3-a456-426614174000",
    }
    assert generate_example(json_schema, json_schema) == expected


def test_no_type():
    """If the schema has no type, it should return None."""
    schema = {}
    assert generate_example(schema, None) is None


def test_anyof_with_invalid_then_valid():
    """Test an 'anyOf' scenario where the first alternative returns None (e.g. unknown type) and the second alternative returns a valid example."""
    schema = {
        "anyOf": [
            {"type": "unknown"},  # this should return None
            {"type": "string", "example": "valid alternative"},
        ]
    }
    assert generate_example(schema, None) == "valid alternative"


def test_array_without_items():
    """For an array with no 'items' defined, it should return a list with a single None."""
    schema = {"type": "array"}
    assert generate_example(schema, None) == [None]


def test_enum_example():
    """For a schema with an enum, it should return default value."""
    schema = {
        "$defs": {
            "RetrieverType": {
                "description": "Type of retriever to use for search.",
                "enum": ["hybrid", "sparse", "dense"],
                "title": "RetrieverType",
                "type": "string",
            }
        },
        "properties": {
            "retriever_type": {"$ref": "#/$defs/RetrieverType", "default": "hybrid"}
        },
        "title": "RequestModel",
        "type": "object",
    }
    expected = {"retriever_type": "hybrid"}
    assert generate_example(schema, schema) == expected
