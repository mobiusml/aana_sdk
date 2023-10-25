from typing import Any, Dict


class AanaException(Exception):
    """
    Base class for SDK exceptions.
    """

    extra = {}

    def __str__(self) -> str:
        """
        Return a string representation of the exception.

        String is defined as follows:
        ```
        <class_name>(extra_key1=extra_value1, extra_key2=extra_value2, ...)
        ```
        """
        class_name = self.__class__.__name__
        extra_str = ""
        for key, value in self.extra.items():
            extra_str += f", {key}={value}"
        return f"{class_name}({extra_str})"

    def get_data(self) -> Dict[str, Any]:
        """
        Get the data to be returned to the client.

        Returns:
            Dict[str, Any]: data to be returned to the client
        """
        data = self.extra.copy()
        return data

    def add_extra(self, key: str, value: Any):
        """
        Add extra data to the exception.

        This data will be returned to the user as part of the response.

        How to use: in the exception handler, add the extra data to the exception and raise it again.

        Example:
            ```
            try:
                ...
            except AanaException as e:
                e.add_extra('extra_key', 'extra_value')
                raise e
            ```

        Args:
            key (str): key of the extra data
            value (Any): value of the extra data
        """
        self.extra[key] = value


class InferenceException(AanaException):
    """Exception raised when there is an error during inference.

    Attributes:
        model_name -- name of the model
    """

    def __init__(self, model_name: str = ""):
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

class MultipleFileUploadNotAllowed(AanaException):
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
