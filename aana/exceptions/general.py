from typing import Any, Dict
from mobius_pipeline.exceptions import BaseException


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
        self.input_name = input_name
        super().__init__()

    def __reduce__(self):
        return (self.__class__, (self.input_name,))
