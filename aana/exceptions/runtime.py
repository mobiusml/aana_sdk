from aana.exceptions.core import BaseException

__all__ = [
    "InferenceException",
    "PromptTooLongException",
    "EndpointNotFoundException",
    "TooManyRequestsException",
    "HandlerAlreadyRegisteredException",
    "HandlerNotRegisteredException",
    "UploadedFileNotFound",
    "DeploymentException",
    "InsufficientResources",
    "FailedDeployment",
    "EmptyMigrationsException",
]


class InferenceException(BaseException):
    """Exception raised when there is an error during inference.

    Attributes:
        model_name -- name of the model
    """

    def __init__(self, model_name: str, message: str | None = None):
        """Initialize the exception.

        Args:
            model_name (str): name of the model that caused the exception
            message (str): the message to display
        """
        message = message or f"Inference failed for model: {model_name}"
        super().__init__(
            model_name=model_name,
            message=message,
        )
        self.model_name = model_name
        self.message = message

    def __reduce__(self):
        """Called when the exception is pickled.

        We need to do this if exception has one or more arguments.
        See https://bugs.python.org/issue32696#msg310963 for more info.
        """
        # TODO: check if there is a better way to do this
        return (self.__class__, (self.model_name, self.message))


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


class EmptyMigrationsException(BaseException):
    """Exception raised when there are no migrations to apply."""

    pass


class DeploymentException(Exception):
    """Base exception for deployment errors."""

    pass


class InsufficientResources(DeploymentException):
    """Exception raised when there are insufficient resources for a deployment."""

    pass


class FailedDeployment(DeploymentException):
    """Exception raised when there is an error during deployment."""

    pass


class UploadedFileNotFound(Exception):
    """Exception raised when the uploaded file is not found."""

    def __init__(self, filename: str):
        """Initialize the exception.

        Args:
            filename (str): the name of the file that was not found
        """
        super().__init__()
        self.filename = filename

    def __reduce__(self):
        """Used for pickling."""
        return (self.__class__, (self.filename,))


class InvalidWebhookEventType(BaseException):
    """Exception raised when an invalid webhook event type is provided."""

    def __init__(self, event_type: str):
        """Initialize the exception.

        Args:
            event_type (str): the invalid event type
        """
        super().__init__(event_type=event_type)
        self.event_type = event_type

    def __reduce__(self):
        """Used for pickling."""
        return (self.__class__, (self.event_type,))
