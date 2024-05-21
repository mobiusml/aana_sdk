from typing import Any, TypeVar

from pydantic import BaseModel, RootModel


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
        """Check if model contains item."""
        return item in self.root

    def __add__(self, other):
        """Add two models."""
        return self.__class__(root=self.root + other.root)


OptionType = TypeVar("OptionType", bound=BaseModel)


def merged_options(default_options: OptionType, options: OptionType) -> OptionType:
    """Merge options into default_options.

    Args:
        default_options (OptionType): default options
        options (OptionType): options to be merged into default_options

    Returns:
        OptionType: merged options
    """
    # if options is None, return default_options
    if options is None:
        return default_options.model_copy()
    # options and default_options have to be of the same type
    if type(default_options) != type(options):
        raise ValueError("Option type mismatch.")  # noqa: TRY003
    default_options_dict = default_options.model_dump()
    for k, v in options.model_dump().items():
        if v is not None:
            default_options_dict[k] = v
    return options.__class__.model_validate(default_options_dict)


def pydantic_to_dict(data: Any) -> Any:
    """Convert all Pydantic objects in the structured data.

    Args:
        data (Any): the structured data

    Returns:
        Any: the same structured data with Pydantic objects converted to dictionaries
    """
    if isinstance(data, BaseModel):
        return data.model_dump()
    elif isinstance(data, list):
        return [pydantic_to_dict(item) for item in data]
    elif isinstance(data, dict):
        return {key: pydantic_to_dict(value) for key, value in data.items()}
    else:
        return data  # return as is for non-Pydantic types
