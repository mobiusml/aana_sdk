import urllib.request
from contextlib import ExitStack
from pathlib import Path

import requests
from tqdm import tqdm

from aana.configs.settings import settings
from aana.exceptions.io import DownloadException
from aana.utils.file import get_sha256_hash_file


# Issue-Enable HF download: https://github.com/mobiusml/aana_sdk/issues/65
# model download from a url and checking SHA sum of the URL.
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
