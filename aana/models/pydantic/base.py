from pydantic import BaseModel
from pyparsing import Any


class BaseListModel(BaseModel):
    """The base model for pydantic models with list as root.

    It makes pydantic models with list as root behave like normal lists.
    """

    def __iter__(self):
        """Get iterator for model."""
        return iter(self.__root__)

    def __len__(self):
        """Get length of model."""
        return len(self.__root__)

    def __getitem__(self, index):
        """Get item at index of model."""
        return self.__root__[index]

    def __setitem__(self, index, value):
        """Set item at index of model."""
        self.__root__[index] = value

    def __delitem__(self, index):
        """Remove item at index of model."""
        del self.__root__[index]

    def __contains__(self, item):
        """Check if modle contains item."""
        return item in self.__root__

    def __add__(self, other):
        """Add two models."""
        return self.__class__(__root__=self.__root__ + other.__root__)


class BaseStringModel(BaseModel):
    """The base model for pydantic models that are just strings."""

    __root__: str

    def __init__(self, __root__value: Any = None, **data):
        """Initialize the model."""
        if __root__value is not None:
            super().__init__(__root__=__root__value, **data)
        else:
            super().__init__(**data)

    def __str__(self) -> str:
        """Convert to a string."""
        return self.__root__

    def __repr__(self) -> str:
        """Convert to a string representation."""
        return f"{self.__class__.__name__}({self.__root__!r})"

    def __eq__(self, other: Any) -> bool:
        """Check if two models are equal."""
        if isinstance(other, self.__class__):
            return self.__root__ == other.__root__
        if isinstance(other, str):
            return self.__root__ == other
        return NotImplemented

    def __hash__(self) -> int:
        """Get hash of model."""
        return hash(self.__root__)

    def __getitem__(self, key):
        """Get item at key of model."""
        return self.__root__[key]

    def __len__(self) -> int:
        """Get length of model."""
        return len(self.__root__)

    def __iter__(self):
        """Get iterator for model."""
        return iter(self.__root__)

    def __contains__(self, item):
        """Check if modle contains item."""
        return item in self.__root__

    def __add__(self, other):
        """Add two models or a model and a string."""
        if isinstance(other, self.__class__):
            return self.__class__(__root__=self.__root__ + other.__root__)
        if isinstance(other, str):
            return str(self.__root__) + other
        return NotImplemented

    def __getattr__(self, item):
        """Automatically delegate method calls to self.__root__ if they are not found in the model.

        Check if the attribute is a callable (method) of __root__ and return a wrapped call if it is.
        This will handle methods like startswith, endswith, and split.
        """
        attr = getattr(self.__root__, item)
        if callable(attr):

            def wrapper(*args, **kwargs):
                return attr(*args, **kwargs)

            return wrapper
        return attr
