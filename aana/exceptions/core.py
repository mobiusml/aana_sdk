from typing import Any


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

    def add_extra(self, data: dict[str, Any]) -> None:
        """Add extra data to the exception.

        This data will be returned to the user as part of the response.

        How to use: in the exception handler, add the extra data to the exception and raise it again.

        Example:
            ```
            try:
                ...
            except BaseException as e:
                e.add_extra({'extra_key': 'extra_value'})
                raise e
            ```

        Args:
            data (dict[str, Any]): dictionary containing the extra data
        """
        self.extra.update(data)
