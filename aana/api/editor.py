import ast
import json
from collections import defaultdict
from glob import glob
from importlib import resources
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
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
    aana_base_path = str(resources.path("aana", ""))
    import_path = file_path.replace(aana_base_path, "aana").replace("/", ".")[:-3]

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
            class_names.append((import_path, node))

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
    classes = [
        (path, c) for path, c in classes if "BaseModel" in [c.id for c in c.bases]
    ]
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
    classes = [c for _, c in classes if "BaseDeployment" in [c.id for c in c.bases]]
    return classes


def get_deployments_typed_dicts() -> list:
    directory_path = resources.path("aana.deployments", "")

    classes = get_class_names_from_directory(directory_path)

    # keep only TypedDict
    classes = [c for _, c in classes if "TypedDict" in [c.id for c in c.bases]]
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


available_functions = [
    "aana.utils.video.download_video",
    "aana.utils.video.extract_frames_decord",
    "aana.utils.video.generate_frames_decord",
    "aana.utils.db.save_video_transcription",
    "aana.utils.general.get_attribute",
    "aana.utils.general.format_prompt_as_chat_dialog",
    "aana.utils.db.save_video",
    "aana.utils.db.delete_media",
    "aana.utils.db.save_video_transcription",
    "aana.utils.db.save_video_captions",
    "aana.utils.db.load_video_metadata",
    "aana.utils.db.load_video_transcription",
    "aana.utils.db.load_video_captions",
    "aana.utils.video.generate_combined_timeline",
    "aana.utils.video.generate_dialog",
]


@app.get("/")
async def index(request: Request):
    available_pipelines = glob(str(resources.path("aana.configs.endpoints", "*.json")))
    # only keep the name of the pipeline
    available_pipelines = [Path(p).stem for p in available_pipelines]

    return templates.TemplateResponse(
        "index.html", {"request": request, "pipelines": available_pipelines}
    )


@app.post("/add_pipeline")
async def add_pipeline(request: Request):
    # get form data
    data = await request.form()
    pipeline_name = data["name"]
    print(pipeline_name)

    pipeline_file = resources.path("aana.configs.pipelines", f"{pipeline_name}.json")
    if not pipeline_file.exists():
        with pipeline_file.open("w") as f:
            f.write("[]")

    endpoints_file = resources.path("aana.configs.endpoints", f"{pipeline_name}.json")
    if not endpoints_file.exists():
        with endpoints_file.open("w") as f:
            f.write("[]")

    return RedirectResponse(url=f"/pipeline/{pipeline_name}", status_code=303)


@app.get("/from_pipeline/{pipeline_name}")
async def get_node_editor_from_pipeline(request: Request, pipeline_name: str):
    """The endpoint for getting the node editor.

    Returns:
        AanaJSONResponse: The response containing the node editor.
    """
    deployment_class_to_name = {
        deployment.name: deployment_name
        for deployment_name, deployment in all_deployments.items()
    }

    pydantic_models = get_pydantic_models()
    pydantic_model_names = [m.name for _, m in pydantic_models]

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
                        for _, c in classes
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

    functions = {}
    for func in available_functions:
        functions[func] = get_function_details(func, outputs)

    # with open("/workspaces/aana_sdk/aana/test_pipeline.json", "r") as f:
    #     pipeline = json.load(f)

    pipeline_file = resources.path("aana.configs.pipelines", f"{pipeline_name}.json")
    if not pipeline_file.exists():
        return HTMLResponse(content="Pipeline not found", status_code=404)

    with pipeline_file.open("r") as f:
        pipeline = json.load(f)

    endpoints_file = resources.path("aana.configs.endpoints", f"{pipeline_name}.json")
    if not endpoints_file.exists():
        return HTMLResponse(content="Endpoints not found", status_code=404)

    with endpoints_file.open("r") as f:
        endpoints = json.load(f)

    output_to_node_id = {}
    for i, node in enumerate(pipeline):
        for output_index, output in enumerate(node["outputs"]):
            output_to_node_id[output["name"]] = {"node_id": i + 1, "slot": output_index}

    last_link_id = 0
    all_links = {
        "output": defaultdict(
            list
        ),  # output name to list of link ids, because multiple nodes can use the same output as input
        "input": {},  # node name + input name to link id, because connection comes from only one output
    }
    links = []
    for node_index, node in enumerate(pipeline):
        for input_index, input in enumerate(node["inputs"]):
            all_links["input"][f"{node['name']}:{input['name']}"] = last_link_id + 1
            all_links["output"][input["name"]].append(last_link_id + 1)

            links.append(
                [
                    last_link_id + 1,  # link id
                    output_to_node_id[input["name"]]["node_id"],  # origin node id
                    output_to_node_id[input["name"]]["slot"],  # origin slot
                    node_index + 1,  # target node id
                    input_index,  # target slot
                    "string",  # type
                ]
            )

            last_link_id += 1

    editor_nodes = []
    last_node_id = len(pipeline)
    for i, node in enumerate(pipeline):
        # last_node_id += 1
        node_id = i + 1
        editor_node = {}
        editor_node["id"] = node_id
        editor_node["pos"] = [node_id * 300, node_id * 300]
        # editor_node["size"] = {"0": 400, "1": 200}
        editor_node["flags"] = {}
        editor_node["order"] = node_id
        editor_node["mode"] = 0
        editor_node["title"] = node["name"]

        if node["type"] == "ray_task" or node["type"] == "function":
            # break

            function_name = node["function"]
            function_def = functions[function_name]

            if function_def["is_generator"] != (node.get("data_type") == "generator"):
                raise ValueError(
                    f"Function {function_name} is a generator but the pipeline node is not or vice versa"
                )

            if node.get("dict_output", True) != function_def["dict_output"]:
                raise ValueError(
                    f"Function {function_name} returns a dict but the pipeline node does not or vice versa"
                )

            editor_node["type"] = "pipeline/function"
            editor_node["properties"] = {
                "name": node["name"],
                "function_name": function_name,
                "is_generator": node.get("data_type") == "generator",
                "dict_output": node.get("dict_output", True),
                "is_ray_task": node["type"] == "ray_task",
                "batched": node.get("batched", False),
            }
            if node.get("data_type") == "generator":
                editor_node["properties"]["generator_path"] = node["generator_path"]

            is_ray_task = node["type"] == "ray_task"
            editor_node["widgets_values"] = [
                node["function"],
                is_ray_task,
                node.get("batched", False),
            ]

        elif node["type"] == "ray_deployment":
            function_def = deployments[node["deployment_name"]][node["method"]]

            editor_node["type"] = "pipeline/deployment"

            editor_node["properties"] = {
                "name": node["name"],
                "deployment_name": node["deployment_name"],
                "method": node["method"],
                "is_generator": node.get("data_type") == "generator",
            }
            if node.get("data_type") == "generator":
                editor_node["properties"]["generator_path"] = node["generator_path"]
            editor_node["widgets_values"] = [
                node["deployment_name"],
                node["method"],
            ]
        elif node["type"] == "input":
            editor_node["type"] = "pipeline/input"
            output_type = node["outputs"][0]["data_model"].split(".")[-1]
            editor_node["properties"] = {
                "type": output_type,
            }
            editor_node["widgets_values"] = [node["name"], output_type]
        else:
            raise ValueError(f"Unknown node type {node['type']}")

        inputs = node["inputs"]
        outputs = node["outputs"]

        input_by_key = {input["key"]: input for input in inputs}
        output_by_key = {output["key"]: output for output in outputs}

        if (
            node["type"] == "ray_task"
            or node["type"] == "function"
            or node["type"] == "ray_deployment"
        ):
            input_defs = {input["name"]: input for input in function_def["inputs"]}
            output_defs = {output["name"]: output for output in function_def["outputs"]}

            for output_key in output_defs:
                path = output_by_key.get(output_key, {}).get("path", "")
                editor_node["properties"][f"path[{output_key}]"] = path
        else:
            input_defs = None
            output_defs = None

        # if node["type"] == "input":
        # output_defs =

        editor_node["inputs"] = []
        for input_index, input in enumerate(inputs):
            if input_defs:
                input_def = input_defs[input["key"]]
                input_key = f"{input_def['name']} [{input_def['type']}]"
                if input_def["default"] is not None:
                    input_key += f" = {input_def['default']}"
            else:
                input_key = input["name"]
            editor_node["inputs"].append(
                {
                    "name": input_key,
                    # "type": input.get("data_model", "Any"),
                    "type": "string",
                    "link": all_links["input"][f"{node['name']}:{input['name']}"],
                }
            )
        # add inputs that are in the function definition but not in the pipeline node
        if input_defs and len(inputs) < len(input_defs):
            for input_key, input_def in input_defs.items():
                if input_key not in [input["key"] for input in inputs]:
                    input_name = f"{input_def['name']} [{input_def['type']}]"
                    if input_def["default"] is not None:
                        input_name += f" = {input_def['default']}"
                    editor_node["inputs"].append(
                        {
                            "name": input_name,
                            "type": "string",
                            "link": None,
                        }
                    )

        editor_node["outputs"] = []
        for output_index, output in enumerate(outputs):
            if node["type"] == "input":
                output_type = node["outputs"][0]["data_model"].split(".")[-1]
                output_name = f"{node['name']} [{output_type}]"

                editor_node["outputs"].append(
                    {
                        "name": output_name,
                        "type": "string",
                        "links": all_links["output"][output["name"]],
                        "slot_index": output_index,
                    }
                )

            else:
                if output_defs:
                    output_def = output_defs[output["key"]]
                    output_name = f"{output_def['name']} [{output_def['type']}]"
                else:
                    output_name = output["name"]
                editor_node["outputs"].append(
                    {
                        "name": output_name,
                        # "type": output.get("data_model", "Any"),
                        "type": "string",
                        "links": all_links["output"][output["name"]],
                        "slot_index": output_index,
                    }
                )

        # add outputs that are in the function definition but not in the pipeline node
        if output_defs and len(outputs) < len(output_defs):
            for output_key, output_def in output_defs.items():
                if output_key not in [output["key"] for output in outputs]:
                    output_name = f"{output_def['name']} [{output_def['type']}]"
                    editor_node["outputs"].append(
                        {
                            "name": output_name,
                            "type": "string",
                            "links": [],
                            "slot_index": None,
                        }
                    )

        # if one of the inputs is partial, then add property to the node to indicate that

        editor_node["properties"]["is_partial"] = False
        for input in inputs:
            if input.get("partial"):
                editor_node["properties"]["is_partial"] = True

        # if node["type"] == "ray_task" or node["type"] == "function" or node["type"] == "ray_deployment":
        #     for output_key in output_defs:
        #         path = output_by_key.get(output_key, {}).get("path", "")
        #         editor_node["properties"][f"path[{output_key}]"] = path

        if "kwargs" in node:
            kwards = node["kwargs"]
            for key, value in kwards.items():
                print(key, value)
                last_node_id += 1
                editor_kwargs_node = {
                    "id": last_node_id,
                    "pos": [last_node_id * 300, last_node_id * 300],
                    "size": {"0": 150, "1": 100},
                    "flags": {},
                    "order": last_node_id,
                    "mode": 0,
                    "title": "Fixed Input",
                    "type": "pipeline/fixed_input",
                    "properties": {
                        "value": value,
                    },
                    "inputs": [],
                    "outputs": [],
                    "widgets_values": [value],
                }
                editor_nodes.append(editor_kwargs_node)

                target_slot = None
                for slot, item in enumerate(editor_node["inputs"]):
                    if item["name"].startswith(f"{key} ["):
                        target_slot = slot
                        item["link"] = last_link_id + 1
                        break
                target_node_id = editor_node["id"]

                links.append(
                    [
                        last_link_id + 1,  # link id
                        editor_kwargs_node["id"],  # origin node id
                        0,  # origin slot
                        target_node_id,  # target node id
                        target_slot,  # target slot
                        "string",  # type
                    ]
                )
                last_link_id += 1

        editor_nodes.append(editor_node)

    for endpoint in endpoints:
        last_node_id += 1
        editor_node = {
            "id": last_node_id + 1,
            "pos": [last_node_id * 300, last_node_id * 300],
            # "size": {"0": 400, "1": 200},
            "flags": {},
            "order": last_node_id + 1,
            "mode": 0,
            "title": endpoint["name"],
            "type": "pipeline/endpoint",
            "properties": {
                "path": endpoint["path"],
                "summary": endpoint["summary"],
                "count_outputs": len(endpoint["outputs"]),
            },
            "inputs": [],
            "outputs": [],
            "widgets_values": [endpoint["path"], endpoint["summary"]],
        }
        for output_index, output in enumerate(endpoint["outputs"]):
            editor_node["properties"][f"output_{output_index}"] = output["name"]
            editor_node["properties"][f"is_streaming_{output_index}"] = output.get(
                "streaming", False
            )
            if output["output"] in output_to_node_id:
                origin_node = output_to_node_id[output["output"]]
                origin_node_obj = {
                    editor_node["id"]: editor_node for editor_node in editor_nodes
                }[origin_node["node_id"]]
                links.append(
                    [
                        last_link_id + 1,  # link id
                        origin_node["node_id"],  # origin node id
                        origin_node["slot"],  # origin slot
                        editor_node["id"],  # target node id
                        output_index,  # target slot
                        "string",  # type
                    ]
                )
                origin_node_obj["outputs"][origin_node["slot"]]["links"].append(
                    last_link_id + 1
                )
                link = last_link_id + 1
                last_link_id += 1
            else:
                link = None

            editor_node["inputs"].append(
                {
                    "name": f"{output['name']} (streaming)"
                    if output.get("streaming")
                    else output["name"],
                    "type": "string",
                    "link": link,
                }
            )

        editor_nodes.append(editor_node)

    editor_pipeline = {
        "nodes": editor_nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
        "last_node_id": len(editor_nodes),
        "last_link_id": last_link_id,
    }

    # return HTMLResponse(content=html)
    return templates.TemplateResponse(
        "editor.html",
        {
            "request": request,
            "data_models": pydantic_model_names,
            "deployments": json.dumps(deployments),
            "functions": json.dumps(functions),
            "pipeline": json.dumps(editor_pipeline),
            "pipeline_name": pipeline_name,
        },
    )


@app.get("/pipeline/{pipeline_name}")
async def get_node_editor(request: Request, pipeline_name: str):
    """The endpoint for getting the node editor.

    Returns:
        AanaJSONResponse: The response containing the node editor.
    """
    deployment_class_to_name = {
        deployment.name: deployment_name
        for deployment_name, deployment in all_deployments.items()
    }

    pydantic_models = get_pydantic_models()
    pydantic_model_names = [m.name for _, m in pydantic_models]

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
                        for _, c in classes
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

    functions = {}
    for func in available_functions:
        functions[func] = get_function_details(func, outputs)

    editor_file = resources.path("aana.configs.editor", f"{pipeline_name}.json")
    with open(editor_file, "r") as f:
        editor_pipeline = json.load(f)

    return templates.TemplateResponse(
        "editor.html",
        {
            "request": request,
            "data_models": pydantic_model_names,
            "deployments": json.dumps(deployments),
            "functions": json.dumps(functions),
            "pipeline": json.dumps(editor_pipeline),
            "pipeline_name": pipeline_name,
        },
    )


@app.post("/save/{pipeline_name}")
async def save_pipeline(request: Request, pipeline_name: str):
    deployment_class_to_name = {
        deployment.name: deployment_name
        for deployment_name, deployment in all_deployments.items()
    }

    pydantic_models = get_pydantic_models()
    pydantic_model_names = [m.name for _, m in pydantic_models]
    pydantic_model_paths = {m.name: f"{path}.{m.name}" for path, m in pydantic_models}
    pydantic_model_paths.update(
        {f"{path}.{m.name}": f"{path}.{m.name}" for path, m in pydantic_models}
    )

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
                        for _, c in classes
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

    functions = {}
    for func in available_functions:
        functions[func] = get_function_details(func, outputs)

    graph = await request.json()
    print(pipeline_name, graph)

    node_id_to_node = {node["id"]: node for node in graph["nodes"]}

    from collections import defaultdict
    from typing import Any

    all_outputs = {}
    output_by_location = defaultdict(dict)
    for node in graph["nodes"]:
        if node["type"] == "pipeline/fixed_input":
            pass

        elif node["type"] == "pipeline/input":
            output_name, output_type, path = node["widgets_values"]

            for slot_index, output in enumerate(node["outputs"]):
                name = output["name"]

                if path == "":
                    path = output_name

                # output_name = node["title"].replace(" ", "_").lower() + "_" + name
                k = 2
                while output_name in all_outputs:
                    output_name = (
                        node["title"].replace(" ", "_").lower() + "_" + name + f"_{k}"
                    )

                output_def = {
                    "name": output_name,
                    "key": output_name,
                    "path": path,
                    "data_model": pydantic_model_paths[output_type],
                }

                all_outputs[output_name] = output_def

                # if "slot_index" in output:
                #     output_by_location[node["id"]][output["slot_index"]] = output_def
                output_by_location[node["id"]][slot_index] = output_def
        else:
            for slot_index, output in enumerate(node.get("outputs", [])):
                name_with_type = output["name"]
                name = name_with_type.split("[")[0].strip()

                # output_name = node["title"].replace(" ", "_").lower() + "_" + name
                if node["type"] == "pipeline/function":
                    function_name = node["properties"]["function_name"].split(".")[-1]
                    output_name = f"{function_name}_{name}"
                else:
                    output_name = name
                k = 2
                while output_name in all_outputs:
                    output_name = (
                        node["title"].replace(" ", "_").lower() + "_" + name + f"_{k}"
                    )

                output_type = name_with_type.split("[")[1].split("]")[0]
                path = node["properties"][f"path[{name}]"]
                if path == "":
                    path = output_name

                output_def = {
                    "name": output_name,
                    "key": name,
                    "path": path,
                    # "data_model": pydantic_model_paths.get(output_type, Any),
                }
                if output_type in pydantic_model_paths:
                    output_def["data_model"] = pydantic_model_paths[output_type]

                all_outputs[output_name] = output_def

                # if "slot_index" in output:
                #     output_by_location[node["id"]][output["slot_index"]] = output_def
                output_by_location[node["id"]][slot_index] = output_def

    # print(all_outputs)

    links = {}
    for link in graph["links"]:
        link_id, origin_node_id, origin_slot, target_node_id, target_slot, _ = link
        links[link_id] = {
            "origin_node": origin_node_id,
            "origin_slot": origin_slot,
            "target_node": target_node_id,
            "target_slot": target_slot,
        }

    kwargs = defaultdict(dict)
    for link_id, link in links.items():
        origin_node = node_id_to_node[link["origin_node"]]
        if origin_node["type"] == "pipeline/fixed_input":
            origin_output = node_id_to_node[link["origin_node"]]["outputs"][
                link["origin_slot"]
            ]
            target_input = node_id_to_node[link["target_node"]]["inputs"][
                link["target_slot"]
            ]

            kwargs_name = target_input["name"].split("[")[0].strip()

            kwargs[link["target_node"]][kwargs_name] = origin_node["properties"][
                "value"
            ]

    from typing import Any

    print(output_by_location)

    pipeline = []
    endpoints = []

    for node in graph["nodes"]:
        if node["type"] == "pipeline/fixed_input":
            continue
        elif node["type"] == "pipeline/input":
            pipeline_node = {
                "name": node["title"].replace("Input: ", "input_"),
                "type": "input",
                "inputs": [],
            }
            # break
            pipeline_node["name"] = output_by_location[node["id"]][0]["name"]
            # output_name, output_type, path = node["widgets_values"]

            # outputs = []
            # for output in node["outputs"]:
            #     outputs.append(
            #         {
            #             "name": output["name"],
            #             "key": output["name"],
            #             "path": path if path else output["name"],
            #             "data_model": output_type,
            #         }
            #     )

            # pipeline_node["outputs"] = outputs
        elif node["type"] == "pipeline/deployment":
            pipeline_node = {
                "name": node["title"].replace(" ", "_"),
                "type": "ray_deployment",
                "deployment_name": node["properties"]["deployment_name"],
                "method": node["properties"]["method"],
            }
        elif node["type"] == "pipeline/function":
            pipeline_node = {
                "name": node["title"].replace(" ", "_"),
                "function": node["properties"]["function_name"],
                "type": "ray_task" if node["properties"]["is_ray_task"] else "function",
                "dict_output": node["properties"]["dict_output"],
                "batched": node["properties"]["batched"],
            }
            # outputs = []
            # for output in node["outputs"]:
            #     name_with_type = output["name"]
            #     name = name_with_type.split("[")[0].strip()
            #     output_type = name_with_type.split("[")[1].split("]")[0]
            #     path = node["properties"][f"path[{name}]"]
            #     if path == "":
            #         path = name
            #     outputs.append(
            #         {
            #             "name": name,
            #             "key": name,
            #             "path": path,
            #             "data_model": pydantic_model_paths.get(output_type, Any),
            #         }
            #     )
            # pipeline_node["outputs"] = outputs
        elif node["type"] == "pipeline/endpoint":
            pipeline_node = None
            endpoint = {
                # "name": node["title"].replace(" ", "_"),
                "name": node["properties"]["path"].replace("/", "_")[1:],
                "path": node["properties"]["path"],
                "summary": node["properties"]["summary"],
                "streaming": False,
            }
            endpoint_outputs = []
            for input_index, input in enumerate(node["inputs"]):
                if input["link"] is None:
                    continue
                link = links[input["link"]]
                output_def = output_by_location[link["origin_node"]][
                    link["origin_slot"]
                ]
                output_name = node["properties"][f"output_{input_index}"]
                is_streaming = node["properties"][f"is_streaming_{input_index}"]
                endpoint_outputs.append(
                    {
                        "name": output_name,
                        "output": output_def["name"],
                        "streaming": is_streaming,
                    }
                )
                if is_streaming:
                    endpoint["streaming"] = True
            endpoint["outputs"] = endpoint_outputs
            endpoints.append(endpoint)
        else:
            # continue
            # break
            raise ValueError(f"Unknown node type: {node['type']}")

        if node["type"] in ["pipeline/function", "pipeline/deployment"]:
            if (
                "flatten_by" in node["properties"]
                and node["properties"]["flatten_by"] != ""
            ):
                pipeline_node["flatten_by"] = node["properties"]["flatten_by"]

            is_partial = node["properties"].get("is_partial", False)

            if node["properties"].get("is_generator", False):
                pipeline_node["data_type"] = "generator"
                pipeline_node["generator_path"] = node["properties"]["generator_path"]

            if node["id"] in kwargs:
                pipeline_node["kwargs"] = kwargs[node["id"]]

            inputs = []
            for input in node["inputs"]:
                if input["link"] is None:
                    continue
                link = links[input["link"]]
                if (
                    link["origin_node"] not in output_by_location
                    or link["origin_slot"]
                    not in output_by_location[link["origin_node"]]
                ):
                    continue
                output_def = output_by_location[link["origin_node"]][
                    link["origin_slot"]
                ]
                key = input["name"].split("[")[0].strip()
                input_def = {
                    "name": output_def["name"],
                    "key": key,
                    "path": output_def["path"],
                }
                if is_partial:
                    input_def["partial"] = True
                inputs.append(input_def)

            pipeline_node["inputs"] = inputs

        if node["type"] in [
            "pipeline/function",
            "pipeline/deployment",
            "pipeline/input",
        ]:
            outputs = []
            for slot_index, output in enumerate(node["outputs"]):
                # output_def = output_by_location[node["id"]][output["slot_index"]]
                # if output["slot_index"] in output_by_location[node["id"]]:
                # if "slot_index" not in output:
                # continue
                # output_def = output_by_location[node["id"]][output["slot_index"]]
                output_def = output_by_location[node["id"]][slot_index]
                outputs.append(output_def)

            pipeline_node["outputs"] = outputs

        if pipeline_node:
            pipeline.append(pipeline_node)

    pipeline_file = resources.path("aana.configs.pipelines", f"{pipeline_name}.json")
    endpoints_file = resources.path("aana.configs.endpoints", f"{pipeline_name}.json")
    editor_file = resources.path("aana.configs.editor", f"{pipeline_name}.json")

    import json

    with pipeline_file.open("w") as f:
        json.dump(pipeline, f, indent=2)

    with endpoints_file.open("w") as f:
        json.dump(endpoints, f, indent=2)

    with editor_file.open("w") as f:
        json.dump(graph, f, indent=2)

    return {"status": "success"}
