import typing


def is_typed_dict(argument: type) -> bool:
    """Checks if a argument is a TypedDict.

    Arguments:
        argument (type): the type to check

    Returns:
        bool: True if the argument type is a TypedDict anf False if it is not.
    """
    return bool(
        argument and getattr(argument, "__orig_bases__", None) == (typing.TypedDict,)
    )
