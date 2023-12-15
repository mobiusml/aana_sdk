from pydantic import BaseModel


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
