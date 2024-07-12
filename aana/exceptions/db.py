from aana.core.models.media import MediaId
from aana.exceptions.core import BaseException
from aana.storage.models.video import Status as VideoStatus


class NotFoundException(BaseException):
    """Raised when an item searched by id is not found."""

    def __init__(self, table_name: str, id: int | MediaId):  # noqa: A002
        """Constructor.

        Args:
            table_name (str): the name of the table being queried.
            id (int | MediaId): the id of the item to be retrieved.
        """
        super().__init__(table=table_name, id=id)
        self.table_name = table_name
        self.id = id
        self.http_status_code = 404

    def __reduce__(self):
        """Used for pickling."""
        return (self.__class__, (self.table_name, self.id))


class MediaIdAlreadyExistsException(BaseException):
    """Raised when a media_id already exists."""

    def __init__(self, table_name: str, media_id: MediaId):
        """Constructor.

        Args:
            table_name (str): the name of the table being queried.
            media_id (MediaId): the id of the item to be retrieved.
        """
        super().__init__(table=table_name, id=media_id)
        self.table_name = table_name
        self.media_id = media_id

    def __reduce__(self):
        """Used for pickling."""
        return (self.__class__, (self.table_name, self.media_id))


class UnfinishedVideoException(BaseException):
    """Exception raised when try to fetch unfinished video.

    Attributes:
        media_id (int | MediaId): The id of video.
        status (VideoStatus): The current video status.
        message (str): The error message.
    """

    def __init__(self, media_id: int | MediaId, status: VideoStatus, message: str):
        """Constructor.

        Args:
            media_id (int | MediaId): The id of video.
            status (VideoStatus): The current video status.
            message (str): The error message.
        """
        super().__init__(media_id=media_id, status=status, message=message)
        self.media_id = media_id
        self.status = status
        self.message = message

    def __reduce__(self):
        """Used for pickling."""
        return (self.__class__, (self.media_id, self.status, self.message))
