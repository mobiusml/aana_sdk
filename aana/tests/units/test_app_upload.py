# ruff: noqa: S101, S113
import io
import json
from typing import TypedDict

import requests
from pydantic import BaseModel, ConfigDict, Field

from aana.api.api_generation import Endpoint
from aana.exceptions.runtime import UploadedFileNotFound


class FileUploadModel(BaseModel):
    """Model for a file upload input."""

    content: str | None = Field(
        None,
        description="The name of the file to upload.",
    )
    _file: bytes | None = None

    def set_files(self, files: dict[str, bytes]):
        """Set files."""
        if self.content:
            if self.content not in files:
                raise UploadedFileNotFound(self.content)
            self._file = files[self.content]

    model_config = ConfigDict(extra="forbid")


class FileUploadEndpointOutput(TypedDict):
    """The output of the file upload endpoint."""

    text: str


class FileUploadEndpoint(Endpoint):
    """File upload endpoint."""

    async def run(self, file: FileUploadModel) -> FileUploadEndpointOutput:
        """Upload a file.

        Args:
            file (FileUploadModel): The file to upload

        Returns:
            FileUploadEndpointOutput: The uploaded file
        """
        file = file._file
        return {"text": file.decode()}


deployments = []

endpoints = [
    {
        "name": "file_upload",
        "path": "/file_upload",
        "summary": "Upload a file",
        "endpoint_cls": FileUploadEndpoint,
    }
]


def test_file_upload_app(create_app):
    """Test the app with a file upload endpoint."""
    aana_app = create_app(deployments, endpoints)

    port = aana_app.port
    route_prefix = ""

    # Check that the server is ready
    response = requests.get(f"http://localhost:{port}{route_prefix}/api/ready")
    assert response.status_code == 200
    assert response.json() == {"ready": True}

    # Test lowercase endpoint
    # data = {"content": "file.txt"}
    data = {"file": {"content": "file.txt"}}
    file = b"Hello world! This is a test."
    files = {"file.txt": io.BytesIO(file)}
    response = requests.post(
        f"http://localhost:{port}{route_prefix}/file_upload",
        data={"body": json.dumps(data)},
        files=files,
    )
    assert response.status_code == 200, response.text
    text = response.json().get("text")
    assert text == file.decode()
