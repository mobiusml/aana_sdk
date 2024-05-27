from typing import TYPE_CHECKING

from aana.exceptions.core import BaseException

if TYPE_CHECKING:
    from aana.core.models.audio import Audio
    from aana.core.models.image import Image
    from aana.core.models.video import Video


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
