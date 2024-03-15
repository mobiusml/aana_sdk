from pydantic import RootModel
from pyparsing import Any


class BaseListModel(RootModel[list]):
    """The base model for pydantic models with list as root.

    It makes pydantic models with list as root behave like normal lists.
    """

    def __iter__(self):
        """Get iterator for model."""
        return iter(self.root)

    def __len__(self):
        """Get length of model."""
        return len(self.root)

    def __getitem__(self, index):
        """Get item at index of model."""
        return self.root[index]

    def __setitem__(self, index, value):
        """Set item at index of model."""
        self.root[index] = value

    def __delitem__(self, index):
        """Remove item at index of model."""
        del self.root[index]

    def __contains__(self, item):
        """Check if modle contains item."""
        return item in self.root

    def __add__(self, other):
        """Add two models."""
        return self.__class__(root=self.root + other.root)


class BaseStringModel(RootModel[str]):
    """The base model for pydantic models that are just strings."""

    root: str

    def __str__(self) -> str:
        """Convert to a string."""
        return self.root

    def __repr__(self) -> str:
        """Convert to a string representation."""
        return f"{self.__class__.__name__}({self.root!r})"

    def __eq__(self, other: Any) -> bool:
        """Check if two models are equal."""
        if isinstance(other, self.__class__):
            return self.root == other.root
        if isinstance(other, str):
            return self.root == other
        return NotImplemented

    def __hash__(self) -> int:
        """Get hash of model."""
        return hash(self.root)

    def __getitem__(self, key):
        """Get item at key of model."""
        return self.root[key]

    def __len__(self) -> int:
        """Get length of model."""
        return len(self.root)

    def __iter__(self):
        """Get iterator for model."""
        return iter(self.root)

    def __contains__(self, item):
        """Check if modle contains item."""
        return item in self.root

    def __add__(self, other):
        """Add two models or a model and a string."""
        if isinstance(other, self.__class__):
            return self.__class__(root=self.root + other.root)
        if isinstance(other, str):
            return str(self.root) + other
        return NotImplemented

    def __getattr__(self, item):
        """Automatically delegate method calls to self.root if they are not found in the model.

        Check if the attribute is a callable (method) of root and return a wrapped call if it is.
        This will handle methods like startswith, endswith, and split.
        """
        attr = getattr(self.root, item)
        if callable(attr):

            def wrapper(*args, **kwargs):
                return attr(*args, **kwargs)

            return wrapper
        return attr
