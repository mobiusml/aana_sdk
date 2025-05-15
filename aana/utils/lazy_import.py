# Adapted from: https://github.com/deepset-ai/haystack
#
#  SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import types
from types import TracebackType
from typing import Any, Optional

from lazy_imports.try_import import _DeferredImportExceptionContextManager

DEFAULT_IMPORT_ERROR_MSG = "Try 'pip install {}'"


# class LazyImport(_DeferredImportExceptionContextManager):
#     """Wrapper on top of lazy_import's _DeferredImportExceptionContextManager.

#     It adds the possibility to customize the error messages.
#     """

#     def __init__(self, message: str = DEFAULT_IMPORT_ERROR_MSG) -> None:
#         """Initialize the context manager."""
#         super().__init__()
#         self.import_error_msg = message

#     def __exit__(
#         self,
#         exc_type: type[Exception] | None = None,
#         exc_value: Exception | None = None,
#         traceback: TracebackType | None = None,
#     ) -> bool | None:
#         """Exit the context manager.

#         Args:
#             exc_type:
#                 Raised exception type. :obj:`None` if nothing is raised.
#             exc_value:
#                 Raised exception object. :obj:`None` if nothing is raised.
#             traceback:
#                 Associated traceback. :obj:`None` if nothing is raised.

#         Returns:
#             :obj:`None` if nothing is deferred, otherwise :obj:`True`.
#             :obj:`True` will suppress any exceptions avoiding them from propagating.

#         """
#         if isinstance(exc_value, ImportError):
#             message = (
#                 f"Failed to import '{exc_value.name}'. {self.import_error_msg.format(exc_value.name)}. "
#                 f"Original error: {exc_value}"
#             )
#             self._deferred = (exc_value, message)
#             return True
#         return None


class LazyImportLoader(types.ModuleType):
    """Lazy importer that delays module import until first attribute access, and provides a custom error message suggestion if the module is not installed.

    Usage:
        np = LazyImportLoader("np", globals(), "numpy", "Install via pip: 'pip install numpy'")
        # Later in code (this will trigger the import)
        print(np.array([1, 2, 3]))
    """

    def __init__(
        self,
        local_name: str,
        parent_module_globals: dict[str, Any],
        module_name: str,
        error_msg_template: str = "Try 'pip install {}'",
    ):
        """Initialize the lazy import loader."""
        super().__init__(module_name)
        self._local_name = local_name
        self._parent_globals = parent_module_globals
        self._module_name = module_name
        self._error_msg = error_msg_template
        self._module: types.ModuleType | None = None

    def _load_module(self) -> types.ModuleType:
        """Load the module, raising an ImportError with a suggestion if it fails."""
        try:
            module = importlib.import_module(self._module_name)
        except ModuleNotFoundError as err:
            # Raise a clearer ImportError with suggestion
            suggestion = self._error_msg.format(self._module_name)
            raise ImportError(  # noqa: TRY003
                f"Failed to import '{self._module_name}'. {suggestion}. Original error: {err}"
            ) from None

        # Cache in parent globals and sys.modules
        self._parent_globals[self._local_name] = module
        sys.modules[self._local_name] = module

        # Populate this lazy loader's __dict__ for direct attribute access
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the module."""
        if self._module is None:
            self._module = self._load_module()
        return getattr(self._module, name)

    def __dir__(self) -> list[str]:
        """List the attributes of the module."""
        if self._module is None:
            self._module = self._load_module()
        return dir(self._module)


import builtins
import importlib
import sys
from types import ModuleType
from typing import Any, List, Tuple

DEFAULT_IMPORT_ERROR_MSG = "Try 'pip install {}'"


class LazyImport:
    """
    Context‐manager that patches builtins.__import__ so that any
    `from mod import name` inside the block returns a proxy for `name`
    and delays the real import until name is first used (or until .check()).
    """

    def __init__(self, message: str = DEFAULT_IMPORT_ERROR_MSG) -> None:
        self.import_error_msg = message
        self._orig_import = None
        # each entry is (module_path, attr_name, proxy_obj)
        self._proxies: List[Tuple[str, str, "_LazyProxy"]] = []

    def __enter__(self) -> "LazyImport":
        self._orig_import = builtins.__import__
        builtins.__import__ = self._lazy_import
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        # restore normal import
        builtins.__import__ = self._orig_import
        # don’t swallow any exceptions
        return False

    def _lazy_import(
        self,
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> ModuleType:
        # if it's a "from name import A, B, …" statement, we intercept
        if fromlist:
            # build a dummy module, and inject proxies for each attr
            dummy = ModuleType(name)
            for attr in fromlist:
                proxy = _LazyProxy(name, attr, self.import_error_msg)
                self._proxies.append((name, attr, proxy))
                setattr(dummy, attr, proxy)
            return dummy

        # otherwise, just forward the import normally
        return self._orig_import(name, globals, locals, fromlist, level)

    def check(self) -> None:
        """
        Pre‐load all of the deferred imports now, raising immediately
        on missing packages or attributes.
        """
        for module_path, attr, proxy in self._proxies:
            proxy._load()


class _LazyProxy:
    """
    A stand‐in for `getattr(import_module(module_path), attr)`.
    On first access or call, it does the real import, then
    replaces itself with the real object.
    """

    __slots__ = ("_module", "_attr", "_msg", "_obj")

    def __init__(self, module: str, attr: str, msg: str):
        self._module = module
        self._attr = attr
        self._msg = msg
        self._obj = None

    def _load(self):
        if self._obj is None:
            try:
                m = importlib.import_module(self._module)
            except ImportError as e:
                raise ImportError(
                    f"Failed to import '{self._module}'. "
                    f"{self._msg.format(self._module)}. Original error: {e}"
                ) from e

            try:
                self._obj = getattr(m, self._attr)
            except AttributeError:
                raise ImportError(
                    f"Module '{self._module}' has no attribute '{self._attr}'"
                )

        return self._obj

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __getattr__(self, name):
        # e.g. AsyncLLMEngine.__name__, AsyncLLMEngine.some_classmethod, etc.
        return getattr(self._load(), name)

    def __repr__(self):
        return f"<LazyProxy {self._module}.{self._attr}>"
