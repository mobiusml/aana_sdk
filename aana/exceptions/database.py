from mobius_pipeline.exceptions import BaseException

from aana.configs.db import media_id_type


class NotFoundException(BaseException):
    """Raised when an item searched by id is not found."""

    def __init__(self, table_name: str, id: int | media_id_type):  # noqa: A002
        """Constructor.

        Args:
            table_name (str): the name of the table being queried.
            id (media_id_type): the id of the item to be retrieved.
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

    def __init__(self, table_name: str, media_id: media_id_type):
        """Constructor.

        Args:
            table_name (str): the name of the table being queried.
            media_id (media_id_type): the id of the item to be retrieved.
        """
        super().__init__(table=table_name, id=media_id)
        self.table_name = table_name
        self.media_id = media_id

    def __reduce__(self):
        """Used for pickling."""
        return (self.__class__, (self.table_name, self.media_id))
