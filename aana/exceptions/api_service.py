from aana.exceptions.core import BaseException


class ApiKeyNotProvided(BaseException):
    """Exception raised when the API key is not provided."""

    def __init__(self):
        """Initialize the exception."""
        self.message = "API key not provided"
        super().__init__(message=self.message)

    def __reduce__(self):
        """Used for pickling."""
        return (self.__class__, ())


class ApiKeyNotFound(BaseException):
    """Exception raised when the API key is not found.

    Attributes:
        key (str): the API key that was not found
    """

    def __init__(self, key: str):
        """Initialize the exception.

        Args:
            key (str): the API key that was not found
        """
        self.key = key
        self.message = f"API key {key} not found"
        super().__init__(key=key, message=self.message)

    def __reduce__(self):
        """Used for pickling."""
        return (self.__class__, (self.key,))


class InactiveSubscription(BaseException):
    """Exception raised when the subscription is inactive (e.g. credits are not available).

    Attributes:
        key (str): the API key with inactive subscription
    """

    def __init__(self, key: str):
        """Initialize the exception.

        Args:
            key (str): the API key with inactive subscription
        """
        self.key = key
        self.message = (
            f"API key {key} has an inactive subscription. Check your credits."
        )
        super().__init__(key=key, message=self.message)

    def __reduce__(self):
        """Used for pickling."""
        return (self.__class__, (self.key,))


class AdminOnlyAccess(BaseException):
    """Exception raised when the user does not have enough permissions."""

    def __init__(self):
        """Initialize the exception."""
        self.message = "Admin only access"
        super().__init__(message=self.message)

    def __reduce__(self):
        """Used for pickling."""
        return (self.__class__, ())


class ApiKeyValidationFailed(BaseException):
    """Exception raised when the API key validation fails."""

    def __init__(self):
        """Initialize the exception."""
        self.message = "API key validation failed"
        super().__init__(message=self.message)

    def __reduce__(self):
        """Used for pickling."""
        return (self.__class__, ())


class ApiKeyExpired(BaseException):
    """Exception raised when the API key is expired."""

    def __init__(self, key: str):
        """Initialize the exception.

        Args:
            key (str): the expired API key
        """
        self.key = key
        self.message = f"API key {key} is expired."
        super().__init__(key=key, message=self.message)

    def __reduce__(self):
        """Used for pickling."""
        return (self.__class__, (self.key,))
