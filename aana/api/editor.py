import ast
import json
from importlib import resources

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from aana.configs.deployments import deployments as all_deployments

app = FastAPI()

static_dir = resources.path("aana.static", "")
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

templates = Jinja2Templates(resources.path("aana.api.templates", ""))

import importlib.util
import inspect
import os
from importlib import resources

# Function to convert an annotation node to a string
from typing import TypedDict

from pydantic import BaseModel


def get_class_names_from_file(file_path):
    class_names = []

    # Read the content of the file
    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()

    # Parse the file content to an AST
    tree = ast.parse(file_content)

    # Walk the AST to find class definitions
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # class_names.append(node.name)
            class_names.append(node)

    return class_names


def get_class_names_from_directory(directory_path):
    class_names = []

    # List all .py files in the given directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".py") and not filename.startswith("__"):
            file_path = os.path.join(directory_path, filename)

            # # Read the content of the file
            # with open(file_path, "r", encoding="utf-8") as file:
            #     file_content = file.read()

            # # Parse the file content to an AST
            # tree = ast.parse(file_content)

            # # Walk the AST to find class definitions
            # for node in ast.walk(tree):
            #     if isinstance(node, ast.ClassDef):
            #         # class_names.append(node.name)
            #         class_names.append(node)
            class_names.extend(get_class_names_from_file(file_path))

    return class_names


def get_pydantic_models() -> list:
    """
    Load all pydantic models from the aana.models.pydantic package

    Returns:
        list: list of pydantic models
    """
    directory_path = resources.path("aana.models.pydantic", "")

    classes = get_class_names_from_directory(directory_path)

    # keep only pydantic models (BaseModel)
    classes = [c for c in classes if "BaseModel" in [c.id for c in c.bases]]
    return classes


def annotation_to_type(annotation_node):
    if annotation_node is None:
        return None
    # Python 3.9+ can use ast.unparse, for earlier versions use astor or manual conversion
    return ast.unparse(annotation_node)


def get_defaults(func):
    defaults = {}
    for i, arg in enumerate(func.args.args[::-1]):
        if len(func.args.defaults) > i:
            default = func.args.defaults[i]
            defaults[arg.arg] = default.value
    return defaults


# Function to process each function definition in the AST
def get_args(func):
    defaults = get_defaults(func)

    args = []
    for arg in func.args.args:
        if arg.arg == "self":
            continue
        # print(f"Argument name: {arg.arg}, type: {annotation_to_type(arg.annotation)}")
        args.append(
            {
                "name": arg.arg,
                "type": annotation_to_type(arg.annotation),
                "default": defaults.get(arg.arg, None),
            }
        )
    return args


def parse_return_type_annotation(node) -> tuple[bool, str]:
    """Parse the return type annotation of a function definition and return tuple:
    - bool: True if the return type is an AsyncGenerator, False otherwise
    - str: the return type annotation
    """
    if isinstance(node, ast.Subscript):
        # Base type (e.g., "AsyncGenerator")
        base_type = (
            node.value.id if isinstance(node.value, ast.Name) else str(node.value)
        )
        # Type arguments
        if isinstance(node.slice, ast.Index):  # Python 3.8 and earlier
            args = [node.slice.value]
        elif isinstance(node.slice, ast.Tuple):  # For multiple arguments in brackets
            args = node.slice.elts
        else:
            args = [node.slice]  # Python 3.9 and later, direct slice access
        args_str = [ast.unparse(arg) for arg in args]
        # return f"{base_type}[{', '.join(args_str)}]"
        # return (base_type, args_str)
        if base_type == "AsyncGenerator" or base_type == "Generator":
            return True, args_str[0]
        else:
            raise ValueError(f"Unexpected base type: {base_type}")
    # return ast.unparse(node)
    return False, ast.unparse(node)


def get_all_deployments():
    directory_path = resources.path("aana.deployments", "")

    classes = get_class_names_from_directory(directory_path)

    # keep only pydantic models (BaseModel)
    classes = [c for c in classes if "BaseDeployment" in [c.id for c in c.bases]]
    return classes


def get_deployments_typed_dicts() -> list:
    directory_path = resources.path("aana.deployments", "")

    classes = get_class_names_from_directory(directory_path)

    # keep only TypedDict
    classes = [c for c in classes if "TypedDict" in [c.id for c in c.bases]]
    return classes


def deployment_output_to_outputs(output) -> list:
    outputs = []
    for item in output.body:
        if isinstance(item, ast.AnnAssign):
            outputs.append(
                {
                    "name": item.target.id,
                    "type": annotation_to_type(item.annotation),
                }
            )
    return outputs


def get_deployment_methods(deployment):
    methods = []
    for f in deployment.body:
        if isinstance(f, ast.AsyncFunctionDef) and f.name != "apply_config":
            args = get_args(f)
            is_generator, return_type = parse_return_type_annotation(f.returns)

            methods.append(
                {
                    "name": f.name,
                    "args": args,
                    "is_generator": is_generator,
                    "return_type": return_type,
                }
            )
    return methods


def get_function_details(func: str, outputs: dict):
    func_path = (
        str(resources.path(".".join(func.split(".")[:-2]), func.split(".")[-2])) + ".py"
    )
    func_name = func.split(".")[-1]

    with open(func_path, "r") as file:
        file_content = file.read()

    tree = ast.parse(file_content)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            func_outputs = {}
            is_generator, return_type = parse_return_type_annotation(node.returns)
            if return_type in outputs:
                func_outputs = outputs[return_type]
                dict_output = True
            else:
                func_outputs = [{"name": "output", "type": return_type}]
                dict_output = False

            return {
                "name": node.name,
                "dict_output": dict_output,
                "is_generator": is_generator,
                "inputs": get_args(node),
                "outputs": func_outputs,
            }


@app.get("/")
async def get_node_editor(request: Request):
    """The endpoint for getting the node editor.

    Returns:
        AanaJSONResponse: The response containing the node editor.
    """
    deployment_class_to_name = {
        deployment.name: deployment_name
        for deployment_name, deployment in all_deployments.items()
    }

    pydantic_models = get_pydantic_models()
    pydantic_model_names = [m.name for m in pydantic_models]

    # outputs = {}
    # output_dicts = get_deployments_typed_dicts()
    # for output in output_dicts:
    #     # print(output.name)
    #     # print(deployment_output_to_outputs(output))
    #     outputs[output.name] = deployment_output_to_outputs(output)
    # walk all the files in the aana and get all the typed dicts
    typed_dicts = []
    for root, dirs, files in os.walk(resources.path("aana", "")):
        for file in files:
            # Check if the file ends with .py
            if file.endswith(".py"):
                path = os.path.join(root, file)
                # print(path)
                classes = get_class_names_from_file(path)
                # keep only TypedDict
                typed_dicts.extend(
                    [
                        c
                        for c in classes
                        if "TypedDict" in [c.id for c in c.bases if hasattr(c, "id")]
                    ]
                )
    outputs = {}
    for output in typed_dicts:
        # print(output.name)
        # print(deployment_output_to_outputs(output))
        outputs[output.name] = deployment_output_to_outputs(output)

    deployments = {}
    for deployment in get_all_deployments():
        deployments[deployment_class_to_name[deployment.name]] = {}
        for method in get_deployment_methods(deployment):
            deployments[deployment_class_to_name[deployment.name]][method["name"]] = {
                "inputs": method["args"],
                "is_generator": method["is_generator"],
                # "return_type": method["return_type"],
                "outputs": outputs.get(method["return_type"], []),
            }

    available_functions = [
        "aana.utils.video.download_video",
        "aana.utils.video.extract_frames_decord",
        "aana.utils.video.generate_frames_decord",
        "aana.utils.db.save_video_transcription",
    ]

    functions = {}
    for func in available_functions:
        functions[func] = get_function_details(func, outputs)

    with open("/workspaces/aana_sdk/aana/test_pipeline.json", "r") as f:
        pipeline = json.load(f)

    # return HTMLResponse(content=html)
    return templates.TemplateResponse(
        "editor.html",
        {
            "request": request,
            "data_models": pydantic_model_names,
            "deployments": json.dumps(deployments),
            "functions": json.dumps(functions),
            "pipeline": json.dumps(pipeline),
        },
    )


@app.post("/save")
async def save_pipeline(request: Request):
    data = await request.json()
    print(data)
    return {"status": "success"}
