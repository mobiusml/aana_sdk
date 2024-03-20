import hashlib
import urllib.request
from contextlib import ExitStack
from pathlib import Path
from typing import Any, TypeVar

import requests
from pydantic import BaseModel
from tqdm import tqdm

from aana.api.api_generation import Endpoint
from aana.configs.endpoints import endpoints as all_endpoints
from aana.configs.settings import settings
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
        return default_options.model_copy()
    # options and default_options have to be of the same type
    if type(default_options) != type(options):
        raise ValueError("Option type mismatch.")  # noqa: TRY003
    default_options_dict = default_options.model_dump()
    for k, v in options.model_dump().items():
        if v is not None:
            default_options_dict[k] = v
    return options.__class__.model_validate(default_options_dict)


def get_sha256_hash_file(filename: Path) -> str:
    """Compute SHA-256 hash of a file without loading it entirely in memory.

    Args:
        filename (Path): Path to the file to be hashed.

    Returns:
        str: SHA-256 hash of the file in hexadecimal format.
    """
    # Create a sha256 hash object
    sha256 = hashlib.sha256()

    # Open the file in binary mode
    with Path.open(filename, "rb") as f:
        # Read and update hash in chunks of 4K
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)

    # Return the hexadecimal representation of the digest
    return sha256.hexdigest()


# Issue-Enable HF download: https://github.com/mobiusml/aana_sdk/issues/65
# model download from a url and cheking SHA sum of the URL.
def download_model(
    url: str, model_hash: str = "", model_path: Path | None = None, check_sum=True
) -> Path:
    """Download a model from a URL.

    Args:
        url (str): the URL of the file to download
        model_hash (str): hash of the model file for checking sha256 hash if checksum is True
        model_path (Path): optional model path where it needs to be downloaded
        check_sum (bool): boolean to mention whether to check SHA-256 sum or not

    Returns:
        Path: the downloaded file path

    Raises:
        DownloadException: Request does not succeed.
    """
    if model_path is None:
        model_dir = settings.model_dir
        if not model_dir.exists():
            model_dir.mkdir(parents=True)
        model_path = model_dir / "model.bin"

    if model_path.exists() and not model_path.is_file():
        raise RuntimeError(f"Not a regular file: {model_path}")  # noqa: TRY003

    if not model_path.exists():
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

    model_sha256_hash = get_sha256_hash_file(model_path)
    if check_sum and model_sha256_hash != model_hash:
        checksum_error = "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
        raise RuntimeError(f"{checksum_error}")

    return model_path


def download_file(url: str) -> bytes:
    """Download a file from a URL.

    Args:
        url (str): the URL of the file to download

    Returns:
        bytes: the file content

    Raises:
        DownloadException: Request does not succeed.
    """
    # TODO: add retries, check status code, etc.: add issue link
    try:
        response = requests.get(url)  # noqa: S113 TODO : add issue link
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
        return data.model_dump()
    elif isinstance(data, list):
        return [pydantic_to_dict(item) for item in data]
    elif isinstance(data, dict):
        return {key: pydantic_to_dict(value) for key, value in data.items()}
    else:
        return data  # return as is for non-Pydantic types


def get_endpoint(target: str, endpoint: str) -> Endpoint:
    """Get endpoint from endpoints config.

    #TODO: make EndpointList a class and make this a method.: add issue link

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


def get_gpu_memory(gpu: int = 0) -> int:
    """Get the total memory of a GPU in bytes."""
    import torch

    return torch.cuda.get_device_properties(gpu).total_memory
