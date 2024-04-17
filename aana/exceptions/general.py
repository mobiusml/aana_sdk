from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aana.models.core.audio import Audio
    from aana.models.core.image import Image
    from aana.models.core.video import Video


class BaseException(Exception):  # noqa: A001
    """Base class for SDK exceptions."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialise Exception."""
        super().__init__()
        self.extra = kwargs

    def __str__(self) -> str:
        """Return a string representation of the exception.

        String is defined as follows:
        ```
        <class_name>(extra_key1=extra_value1, extra_key2=extra_value2, ...)
        ```
        """
        class_name = self.__class__.__name__
        extra_str_list = []
        for key, value in self.extra.items():
            extra_str_list.append(f"{key}={value}")
        extra_str = ", ".join(extra_str_list)
        return f"{class_name}({extra_str})"

    def get_data(self) -> dict[str, Any]:
        """Get the data to be returned to the client.

        Returns:
            Dict[str, Any]: data to be returned to the client
        """
        data = self.extra.copy()
        return data

    def add_extra(self, key: str, value: Any) -> None:
        """Add extra data to the exception.

        This data will be returned to the user as part of the response.

        How to use: in the exception handler, add the extra data to the exception and raise it again.

        Example:
            ```
            try:
                ...
            except BaseException as e:
                e.add_extra('extra_key', 'extra_value')
                raise e
            ```

        Args:
            key (str): key of the extra data
            value (Any): value of the extra data
        """
        self.extra[key] = value


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


class ImageReadingException(BaseException):
    """Exception raised when there is an error reading an image.

    Attributes:
        image (Image): the image that caused the exception
    """

    def __init__(self, image: "Image"):
        """Initialize the exception.

        Args:
            image (Image): the image that caused the exception
        """
        super().__init__(image=image)
        self.image = image

    def __reduce__(self):
        """Used for pickling."""
        return (self.__class__, (self.image,))


class AudioReadingException(BaseException):
    """Exception raised when there is an error reading an audio.

    Attributes:
        audio (Audio): the audio that caused the exception
    """

    def __init__(self, audio: "Audio"):
        """Initialize the exception.

        Args:
            audio (Audio): the audio that caused the exception
        """
        super().__init__(audio=audio)
        self.audio = audio

    def __reduce__(self):
        """Used for pickling."""
        return (self.__class__, (self.audio,))


class DownloadException(BaseException):
    """Exception raised when there is an error downloading a file.

    Attributes:
        url (str): the URL of the file that caused the exception
    """

    def __init__(self, url: str, msg: str = ""):
        """Initialize the exception.

        Args:
            url (str): the URL of the file that caused the exception
            msg (str): the error message
        """
        super().__init__(url=url)
        self.url = url
        self.msg = msg

    def __reduce__(self):
        """Used for pickling."""
        return (self.__class__, (self.url, self.msg))


class VideoException(BaseException):
    """Exception raised when working with videos.

    Attributes:
        video (Video): the video that caused the exception
    """

    def __init__(self, video: "Video"):
        """Initialize the exception.

        Args:
            video (Video): the video that caused the exception
        """
        super().__init__(video=video)
        self.video = video

    def __reduce__(self):
        """Used for pickling."""
        return (self.__class__, (self.video,))


class VideoReadingException(VideoException):
    """Exception raised when there is an error reading a video.

    Attributes:
        video (Video): the video that caused the exception
    """

    pass


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
