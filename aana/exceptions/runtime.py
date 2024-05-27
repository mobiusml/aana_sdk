from aana.exceptions.core import BaseException


class InferenceException(BaseException):
    """Exception raised when there is an error during inference.

    Attributes:
        model_name -- name of the model
    """

    def __init__(self, model_name):
        """Initialize the exception.

        Args:
            model_name (str): name of the model that caused the exception
        """
        super().__init__(model_name=model_name)
        self.model_name = model_name

    def __reduce__(self):
        """Called when the exception is pickled.

        We need to do this if exception has one or more arguments.
        See https://bugs.python.org/issue32696#msg310963 for more info.
        """
        # TODO: check if there is a better way to do this
        return (self.__class__, (self.model_name,))


class MultipleFileUploadNotAllowed(BaseException):
    """Exception raised when multiple inputs require file upload.

    Attributes:
        input_name -- name of the input
    """

    def __init__(self, input_name: str):
        """Initialize the exception.

        Args:
            input_name (str): name of the input that caused the exception
        """
        super().__init__(input_name=input_name)
        self.input_name = input_name

    def __reduce__(self):
        """Used for pickling."""
        return (self.__class__, (self.input_name,))


class PromptTooLongException(BaseException):
    """Exception raised when the prompt is too long.

    Attributes:
        prompt_len (int): the length of the prompt in tokens
        max_len (int): the maximum allowed length of the prompt in tokens
    """

    def __init__(self, prompt_len: int, max_len: int):
        """Initialize the exception.

        Args:
            prompt_len (int): the length of the prompt in tokens
            max_len (int): the maximum allowed length of the prompt in tokens
        """
        super().__init__(prompt_len=prompt_len, max_len=max_len)
        self.prompt_len = prompt_len
        self.max_len = max_len

    def __reduce__(self):
        """Used for pickling."""
        return (self.__class__, (self.prompt_len, self.max_len))


class EndpointNotFoundException(BaseException):
    """Exception raised when an endpoint is not found.

    Attributes:
        target (str): the name of the target deployment
        endpoint (str): the endpoint path
    """

    def __init__(self, target: str, endpoint: str):
        """Initialize the exception.

        Args:
            target (str): the name of the target deployment
            endpoint (str): the endpoint path
        """
        super().__init__(target=target, endpoint=endpoint)
        self.target = target
        self.endpoint = endpoint

    def __reduce__(self):
        """Used for pickling."""
        return (self.__class__, (self.target, self.endpoint))


class TooManyRequestsException(BaseException):
    """Exception raised when calling a rate-limited resource too often.

    Attributes:
        rate_limit (int): The limit amount.
        rate_duration (float): The duration for the limit in seconds.
    """

    def __init__(self, rate_limit: int, rate_duration: float):
        """Constructor.

        Args:
            rate_limit (int): The limit amount.
            rate_duration (float): The duration for the limit in seconds.
        """
        super().__init__(rate_limit=rate_limit, rate_duration=rate_duration)
        self.rate_limit = rate_limit
        self.rate_duration = rate_duration

    def __reduce__(self):
        """Used for pickling."""
        return (self.__class__, (self.rate_limit, self.rate_duration))


class HandlerAlreadyRegisteredException(BaseException):
    """Exception raised when registering a handler that is already registered."""

    def __init__(self):
        """Constructor."""
        super().__init__()

    def __reduce__(self):
        """Used for pickling."""
        return (self.__class__, ())


class HandlerNotRegisteredException(BaseException):
    """Exception removing a handler that has not been registered."""

    def __init__(self):
        """Constructor."""
        super().__init__()

    def __reduce__(self):
        """Used for pickling."""
        return (self.__class__, ())
