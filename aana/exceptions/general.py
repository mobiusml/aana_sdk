from typing import Any, Dict, Type
from mobius_pipeline.exceptions import BaseException
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aana.models.core.image import Image


class InferenceException(BaseException):
    """Exception raised when there is an error during inference.

    Attributes:
        model_name -- name of the model
    """

    def __init__(self, model_name):
        """
        Initialize the exception.

        Args:
            model_name (str): name of the model that caused the exception
        """
        super().__init__()
        self.model_name = model_name
        self.extra["model_name"] = model_name

    def __reduce__(self):
        # This method is called when the exception is pickled
        # We need to do this if exception has one or more arguments
        # See https://bugs.python.org/issue32696#msg310963 for more info
        # TODO: check if there is a better way to do this
        return (self.__class__, (self.model_name,))


class MultipleFileUploadNotAllowed(BaseException):
    """
    Exception raised when multiple inputs require file upload.

    Attributes:
        input_name -- name of the input
    """

    def __init__(self, input_name: str):
        """
        Initialize the exception.

        Args:
            input_name (str): name of the input that caused the exception
        """
        super().__init__()
        self.input_name = input_name
        self.extra["input_name"] = input_name

    def __reduce__(self):
        return (self.__class__, (self.input_name,))


class ImageReadingException(BaseException):
    """
    Exception raised when there is an error reading an image.

    Attributes:
        image (Image): the image that caused the exception
    """

    def __init__(self, image: "Image"):
        """
        Initialize the exception.

        Args:
            image (Image): the image that caused the exception
        """
        super().__init__()
        self.image = image
        self.extra["image"] = image

    def __reduce__(self):
        return (self.__class__, (self.image,))


class DownloadException(BaseException):
    """
    Exception raised when there is an error downloading a file.

    Attributes:
        url (str): the URL of the file that caused the exception
    """

    def __init__(self, url: str):
        """
        Initialize the exception.

        Args:
            url (str): the URL of the file that caused the exception
        """
        super().__init__()
        self.url = url
        self.extra["url"] = url

    def __reduce__(self):
        return (self.__class__, (self.url,))
