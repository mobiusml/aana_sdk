import pickle
from typing import Any, TypeVar

OptionType = TypeVar("OptionType")


def load_options(s: str, ignore_errors: bool = True) -> Any:
    """
    Load options from a string.

    The string is assumed to be a pickled object encoded in latin1.
    If the string cannot be unpickled, return the string itself if ignore_errors is True,
    otherwise raise an exception.

    The function is used to pass options using Ray's user_config.
    user_config accepts only JSON serializable objects, so we need to encode the options.

    Args:
        s (str): string to be unpickled
        ignore_errors (bool): if True, return the string itself if it cannot be unpickled, otherwise raise an exception

    Returns:z
        unpickled object or the string itself if ignore_errors is True
    """
    try:
        b = s.encode("latin1")
        return pickle.loads(b)
    except Exception as e:
        if ignore_errors:
            return s
        raise e


def encode_options(options: Any) -> str:
    """
    Encode options as a string.

    The string is a pickled object encoded in latin1.

    The function is used to pass options using Ray's user_config.
    user_config accepts only JSON serializable objects, so we need to encode the options.

    Args:
        options (Any): options to be encoded

    Returns:
        str: options encoded as a string
    """
    b = pickle.dumps(options)
    return b.decode("latin1")


def merged_options(default_options: OptionType, options: OptionType) -> OptionType:
    """
    Merge options into default_options.

    Args:
        default_options (OptionType): default options
        options (OptionType): options to be merged into default_options

    Returns:
        OptionType: merged options
    """
    # if options is None, return default_options
    if options is None:
        return default_options
    # options and default_options have to be of the same type
    assert type(default_options) == type(options)
    default_options_dict = default_options.dict()
    for k, v in options.dict().items():
        if v is not None:
            default_options_dict[k] = v
    return options.__class__(**default_options_dict)
