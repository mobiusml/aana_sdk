# ruff: noqa: S101
from collections.abc import Callable

import pytest
from typing_extensions import override

from aana.api.event_handlers.event_handler import EventHandler
from aana.api.event_handlers.event_manager import EventManager
from aana.exceptions.runtime import (
    HandlerNotRegisteredException,
)


class CallbackHandler(EventHandler):
    """A test event handler that just invokes a callback function."""

    def __init__(self, callback: Callable):
        """Constructor."""
        super().__init__()
        self.callback = callback

    @override
    def handle(self, event_name: str, *args, **kwargs):
        self.callback(event_name, *args, **kwargs)


def test_event_dispatch():
    """Tests that event dispatch works correctly."""
    event_manager = EventManager()
    expected_event_name = "foo"
    expected_args = (1, 2, 3, 4, 5)
    expected_kwargs = {"a": "A", "b": "B"}

    def callback(event_name, *args, **kwargs):
        assert event_name == expected_event_name
        assert args == expected_args
        assert kwargs == expected_kwargs

    event_manager.register_handler_for_events(CallbackHandler(callback), ["foo"])

    event_manager.handle("foo", 1, 2, 3, 4, 5, a="A", b="B")


def test_remove_all_raises():
    """Tests that removing handler not added from all events raises an error."""
    event_manager = EventManager()
    handler = CallbackHandler(lambda _, *_args, **_kwargs: None)
    with pytest.raises(HandlerNotRegisteredException):
        event_manager.deregister_handler_from_all_events(handler)


def test_remove_works():
    """Tests that removing a handler works."""
    event_manager = EventManager()
    handler = CallbackHandler(lambda _, *_args, **_kwargs: None)
    event_manager.register_handler_for_events(handler, ["foo"])
    event_manager.deregister_handler_from_event(handler, "foo")
    assert len(event_manager._handlers["foo"]) == 0


def test_remove_all_works():
    """Tests that removing all handlers works."""
    event_manager = EventManager()
    handler = CallbackHandler(lambda _, *_args, **_kwargs: None)
    event_manager.register_handler_for_events(handler, ["foo"])
    event_manager.deregister_handler_from_all_events(handler)
    assert len(event_manager._handlers["foo"]) == 0
