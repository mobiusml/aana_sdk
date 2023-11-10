import io
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError, root_validator, validator
from pydantic.error_wrappers import ErrorWrapper

from aana.models.core.image import Image
from aana.models.pydantic.base import BaseListModel


class ImageInput(BaseModel):
    """
    An image input.

    Exactly one of 'path', 'url', or 'content' must be provided.

    If 'content' or 'numpy' is set to 'file',
    the image will be loaded from the files uploaded to the endpoint.


    Attributes:
        path (str): the file path of the image
        url (str): the URL of the image
        content (bytes): the content of the image in bytes
        numpy (bytes): the image as a numpy array
    """

    path: Optional[str] = Field(None, description="The file path of the image.")
    url: Optional[str] = Field(
        None, description="The URL of the image."
    )  # TODO: validate url
    content: Optional[bytes] = Field(
        None,
        description=(
            "The content of the image in bytes. "
            "Set this field to 'file' to upload files to the endpoint."
        ),
    )
    numpy: Optional[bytes] = Field(
        None,
        description=(
            "The image as a numpy array. "
            "Set this field to 'file' to upload files to the endpoint."
        ),
    )

    def set_file(self, file: bytes):
        """
        If 'content' or 'numpy' is set to 'file',
        the image will be loaded from the file uploaded to the endpoint.

        set_file() should be called after the files are uploaded to the endpoint.

        Args:
            file (bytes): the file uploaded to the endpoint

        Raises:
            ValueError: if the content or numpy isn't set to 'file'
        """
        if self.content == b"file":
            self.content = file
        elif self.numpy == b"file":
            self.numpy = file
        else:
            raise ValueError(
                "The content or numpy of the image must be 'file' to set files."
            )

    def set_files(self, files: List[bytes]):
        """
        Set the files for the image.

        Args:
            files (List[bytes]): the files uploaded to the endpoint

        Raises:
            ValidationError: if the number of images and files aren't the same
        """
        if len(files) != 1:
            error = ErrorWrapper(
                ValueError("The number of images and files must be the same."),
                loc=("images",),
            )
            raise ValidationError([error], self.__class__)
        self.set_file(files[0])

    @root_validator
    def check_only_one_field(cls, values: Dict) -> Dict:
        """
        Check that exactly one of 'path', 'url', 'content' or 'numpy' is provided.

        Args:
            values (Dict): the values of the fields

        Returns:
            Dict: the values of the fields

        Raises:
            ValueError: if not exactly one of 'path', 'url', 'content' or 'numpy' is provided
        """
        count = sum(
            value is not None
            for key, value in values.items()
            if key in ["path", "url", "content", "numpy"]
        )
        if count != 1:
            raise ValueError(
                "Exactly one of 'path', 'url', 'content' or 'numpy' must be provided."
            )
        return values

    def convert_input_to_object(self) -> Image:
        """
        Convert the image input to an image object.

        Returns:
            Image: the image object corresponding to the image input

        Raises:
            ValueError: if the numpy file isn't set
        """
        if self.numpy and self.numpy != b"file":
            try:
                numpy = np.load(io.BytesIO(self.numpy), allow_pickle=False)
            except ValueError:
                raise ValueError("The numpy file isn't valid.")
        elif self.numpy == b"file":
            raise ValueError("The numpy file isn't set. Call set_files() to set it.")
        else:
            numpy = None

        return Image(
            path=Path(self.path) if self.path else None,
            url=self.url,
            content=self.content,
            numpy=numpy,
        )

    class Config:
        schema_extra = {
            "description": (
                "An image. \n"
                "Exactly one of 'path', 'url', or 'content' must be provided. \n"
                "If 'path' is provided, the image will be loaded from the path. \n"
                "If 'url' is provided, the image will be downloaded from the url. \n"
                "The 'content' will be loaded automatically "
                "if files are uploaded to the endpoint (should be set to 'file' for that)."
            )
        }
        validate_assignment = True
        file_upload = True
        file_upload_description = "Upload image file."


class ImageListInput(BaseListModel):
    """
    A pydantic model for a list of images to be used as input.

    Only used for the requests, DO NOT use it for anything else.
    Convert it to a list of image objects with convert_input_to_object().
    """

    __root__: List[ImageInput]

    @validator("__root__", pre=True)
    def check_non_empty(cls, v: List[ImageInput]) -> List[ImageInput]:
        """
        Check that the list of images isn't empty.

        Args:
            v (List[ImageInput]): the list of images

        Returns:
            List[ImageInput]: the list of images

        Raises:
            ValueError: if the list of images is empty
        """
        if len(v) == 0:
            raise ValueError("The list of images must not be empty.")
        return v

    def set_files(self, files: List[bytes]):
        """
        Set the files for the images.

        Args:
            files (List[bytes]): the files uploaded to the endpoint

        Raises:
            ValidationError: if the number of images and files aren't the same
        """

        if len(self.__root__) != len(files):
            error = ErrorWrapper(
                ValueError("The number of images and files must be the same."),
                loc=("images",),
            )
            raise ValidationError([error], self.__class__)
        for image, file in zip(self.__root__, files):
            image.set_file(file)

    def convert_input_to_object(self) -> List[Image]:
        """
        Convert the list of image inputs to a list of image objects.

        Returns:
            List[Image]: the list of image objects corresponding to the image inputs
        """
        return [image.convert_input_to_object() for image in self.__root__]

    class Config:
        schema_extra = {
            "description": (
                "A list of images. \n"
                "Exactly one of 'path', 'url', or 'content' must be provided for each image. \n"
                "If 'path' is provided, the image will be loaded from the path. \n"
                "If 'url' is provided, the image will be downloaded from the url. \n"
                "The 'content' will be loaded automatically "
                "if files are uploaded to the endpoint (should be set to 'file' for that)."
            )
        }
        file_upload = True
        file_upload_description = "Upload image files."
