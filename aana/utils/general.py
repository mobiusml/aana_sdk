import hashlib
import urllib.request
from contextlib import ExitStack
from pathlib import Path
from typing import Any, TypeVar

import requests
import torch
from pydantic import BaseModel
from tqdm import tqdm

from aana.api.api_generation import Endpoint
from aana.configs.endpoints import endpoints as all_endpoints
from aana.exceptions.general import DownloadException, EndpointNotFoundException
from aana.utils.json import jsonify

OptionType = TypeVar("OptionType", bound=BaseModel)


def merged_options(default_options: OptionType, options: OptionType) -> OptionType:
    """Merge options into default_options.

    Args:
        default_options (OptionType): default options
        options (OptionType): options to be merged into default_options

    Returns:
        OptionType: merged options
    """
    # if options is None, return default_options
    if options is None:
        return default_options.copy()
    # options and default_options have to be of the same type
    if type(default_options) != type(options):
        raise ValueError("Option type mismatch.")  # noqa: TRY003
    default_options_dict = default_options.dict()
    for k, v in options.dict().items():
        if v is not None:
            default_options_dict[k] = v
    return options.__class__.parse_obj(default_options_dict)


# model download from a url and cheking SHA sum of the URL.
# TODO: To modify the download function to download model from HF as well.
def download_model(url: str, model_path: Path | None = None) -> Path:
    """Download a model from a URL.

    Args:
        url (str): the URL of the file to download
        model_path (Path): optional model path where it needs to be downloaded

    Returns:
        Path: the downloaded file path

    Raises:
        DownloadException: Request does not succeed.
    """
    if model_path is None:
        model_dir = torch.hub._get_torch_home()
        if not Path(model_dir).exists():
            Path(model_dir).mkdir(parents=True)
        model_path = Path(model_dir) / "pytorch_model.bin"

    if Path(model_path).exists() and not Path(model_path).is_file():
        raise RuntimeError(f"{model_path}")  # exists and is not a regular file

    if not Path(model_path).is_file():
        try:
            with ExitStack() as stack:
                source = stack.enter_context(urllib.request.urlopen(url))  # noqa: S310
                output = stack.enter_context(Path.open(model_path, "wb"))

                loop = tqdm(
                    total=int(source.info().get("Content-Length")),
                    ncols=80,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                )

                with loop:
                    while True:
                        buffer = source.read(8192)
                        if not buffer:
                            break

                        output.write(buffer)
                        loop.update(len(buffer))
        except Exception as e:
            raise DownloadException(url) from e

    model_bytes = Path.open(model_path, "rb").read()

    if hashlib.sha256(model_bytes).hexdigest() != str(url).split("/")[-2]:
        checksum_error = "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
        raise RuntimeError(f"{checksum_error}")

    return Path(model_path)


def download_file(url: str) -> bytes:
    """Download a file from a URL.

    Args:
        url (str): the URL of the file to download

    Returns:
        bytes: the file content

    Raises:
        DownloadException: Request does not succeed.
    """
    # TODO: add retries, check status code, etc.
    try:
        response = requests.get(url)  # noqa: S113 TODO
    except Exception as e:
        raise DownloadException(url) from e
    return response.content


def pydantic_to_dict(data: Any) -> Any:
    """Convert all Pydantic objects in the structured data.

    Args:
        data (Any): the structured data

    Returns:
        Any: the same structured data with Pydantic objects converted to dictionaries
    """
    if isinstance(data, BaseModel):
        return data.dict()
    elif isinstance(data, list):
        return [pydantic_to_dict(item) for item in data]
    elif isinstance(data, dict):
        return {key: pydantic_to_dict(value) for key, value in data.items()}
    else:
        return data  # return as is for non-Pydantic types


def get_endpoint(target: str, endpoint: str) -> Endpoint:
    """Get endpoint from endpoints config.

    #TODO: make EndpointList a class and make this a method.

    Args:
        target (str): the name of the target deployment
        endpoint (str): the endpoint path

    Returns:
        Endpoint: the endpoint

    Raises:
        EndpointNotFoundException: If the endpoint is not found
    """
    for e in all_endpoints[target]:
        if e.path == endpoint:
            return e
    raise EndpointNotFoundException(target=target, endpoint=endpoint)


def get_object_hash(obj: Any) -> str:
    """Get the MD5 hash of an object.

    Objects are converted to JSON strings before hashing.

    Args:
        obj (Any): the object to hash

    Returns:
        str: the MD5 hash of the object
    """
    return hashlib.md5(
        jsonify(obj).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
