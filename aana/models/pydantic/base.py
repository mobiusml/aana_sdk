from pydantic import BaseModel


class BaseListModel(BaseModel):
    """
    The base model for pydantic models with list as root.

    It makes pydantic models with list as root behave like normal lists.
    """

    def __iter__(self):
        return iter(self.__root__)

    def __len__(self):
        return len(self.__root__)

    def __getitem__(self, index):
        return self.__root__[index]

    def __setitem__(self, index, value):
        self.__root__[index] = value

    def __delitem__(self, index):
        del self.__root__[index]

    def __contains__(self, item):
        return item in self.__root__
