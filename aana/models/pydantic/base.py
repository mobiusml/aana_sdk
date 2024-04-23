from pydantic import RootModel


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
