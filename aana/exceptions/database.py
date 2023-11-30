from mobius_pipeline.exceptions import BaseException

from aana.configs.db import id_type


class NotFoundException(BaseException):
    """Raised when an item searched by id is not found."""

    def __init__(self, table_name: str, id: id_type):  # noqa: A002
        """Constructor.

        Arguments:
            table_name (str): the name of the table being queried.
            id (id_type): the id of the item to be retrieved.
        """
        super().__init__(table=table_name, id=id)
