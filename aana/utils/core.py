import hashlib
import importlib
from pathlib import Path
from typing import Any

from aana.utils.json import jsonify


def get_object_hash(obj: Any) -> str:
    """Get the MD5 hash of an object.

    Objects are converted to JSON strings before hashing.

    Args:
        obj (Any): the object to hash

    Returns:
        str: the MD5 hash of the object
    """
    return hashlib.md5(
        jsonify(obj).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()


def import_from(module: str, name, reload=False):
    """Import Module by name."""
    module: Any = __import__(module, fromlist=[str(name)])

    if reload:
        module_path = module.__file__

        # Get the modification timestamp of the module file
        module_last_modified = Path(module_path).stat().st_mtime

        # Check if the module has been imported before
        if hasattr(module, "__last_modified"):
            # Compare the timestamps
            if module_last_modified > module.__last_modified:
                importlib.reload(module)
        else:
            importlib.reload(module)

        # Update the last modified timestamp
        module.__last_modified = module_last_modified

    # return getattr(module, name)
    # getattr doesn't give proper exception, just AttributeError
    # so we use this instead
    return module.__dict__[name]


def import_from_path(path, reload=False):
    """Import a module from path separated by dots.

    :param path: path to the function or class
    :param reload: if True, reload the module
    :return: imported module
    """
    if ":" in path:
        module, name = path.split(":")
    else:
        module, name = ".".join(path.split(".")[:-1]), path.split(".")[-1]
    return import_from(module, name, reload=reload)
