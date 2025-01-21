# Adapted from: https://github.com/deepset-ai/haystack
#
#  SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from types import TracebackType

from lazy_imports.try_import import _DeferredImportExceptionContextManager

DEFAULT_IMPORT_ERROR_MSG = "Try 'pip install {}'"


class LazyImport(_DeferredImportExceptionContextManager):
    """Wrapper on top of lazy_import's _DeferredImportExceptionContextManager.

    It adds the possibility to customize the error messages.
    """

    def __init__(self, message: str = DEFAULT_IMPORT_ERROR_MSG) -> None:
        """Initialize the context manager."""
        super().__init__()
        self.import_error_msg = message

    def __exit__(
        self,
        exc_type: type[Exception] | None = None,
        exc_value: Exception | None = None,
        traceback: TracebackType | None = None,
    ) -> bool | None:
        """Exit the context manager.

        Args:
            exc_type:
                Raised exception type. :obj:`None` if nothing is raised.
            exc_value:
                Raised exception object. :obj:`None` if nothing is raised.
            traceback:
                Associated traceback. :obj:`None` if nothing is raised.

        Returns:
            :obj:`None` if nothing is deferred, otherwise :obj:`True`.
            :obj:`True` will suppress any exceptions avoiding them from propagating.

        """
        if isinstance(exc_value, ImportError):
            message = (
                f"Failed to import '{exc_value.name}'. {self.import_error_msg.format(exc_value.name)}. "
                f"Original error: {exc_value}"
            )
            self._deferred = (exc_value, message)
            return True
        return None
