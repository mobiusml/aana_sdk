import re
from enum import Enum


class VideoSource(str, Enum):
    """Video sources.

    Possible values are "auto" and "youtube".

    Attributes:
        AUTO (str): auto
        YOUTUBE (str): youtube
    """

    AUTO = "auto"
    YOUTUBE = "youtube"

    @classmethod
    def from_url(cls, url: str) -> "VideoSource":
        """Get the video source from a URL.

        Args:
            url (str): the URL

        Returns:
            VideoSource: the video source
        """
        # TODO: Check that the URL is valid

        youtube_pattern = r"^(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/watch\?v=|youtube\.[a-zA-Z]{2,3}(\.[a-zA-Z]{2})?\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)(?:&[^\s]+)*$"

        if re.match(youtube_pattern, url):
            return cls.YOUTUBE
        else:
            return cls.AUTO
