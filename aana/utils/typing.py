import typing


def is_typed_dict(argument: type) -> bool:
    """Checks if a argument is a TypedDict.

    Arguments:
        argument (type): the type to check

    Returns:
        bool: True if the argument type is a TypedDict anf False if it is not.
    """
    return bool(argument and getattr(argument, "__orig_bases__", None) == typing.TypedDict)


def as_dict_of_types(argument: type[typing._TypedDictMeta]) -> dict[str, type]:
    """Extracts a dictionary of field names and types from the type of a TypedDict.

    Arguments:
        argument (type): a type that is a subclass of TypedDict

    Returns:
        dict[str, type]: a dict whose keys are field names and whose values are the types associated with those fields
    """
    return typing.get_type_hints(argument)
