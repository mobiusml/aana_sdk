# ruff: noqa: S101
import pytest

from aana.models.pydantic.base import BaseStringModel


def test_str_model_creation():
    """Test that a media id can be created."""
    str_model = BaseStringModel(__root__="foo")
    assert str_model == "foo"

    str_model = BaseStringModel("foo")
    assert str_model == "foo"

    with pytest.raises(ValueError):
        str_model = BaseStringModel()


def test_str_model_string_behaviour():
    """Test that media id behaves like a string."""
    str_model = BaseStringModel("abc")
    assert str_model.startswith("a")
    assert str_model.endswith("c")
    assert str_model.upper() == "ABC"
    assert str_model.lower() == "abc"
    assert str_model.capitalize() == "Abc"
    assert str_model.title() == "Abc"
    assert str_model.strip() == "abc"
    assert str_model.lstrip() == "abc"
    assert str_model.rstrip() == "abc"
    assert str_model.replace("a", "b") == "bbc"
    assert str_model.count("a") == 1
    assert str_model.index("a") == 0
    assert str_model.find("a") == 0
    assert str_model.rfind("a") == 0
    assert str_model.isalnum()
    assert str_model.isalpha()
    assert not str_model.isdigit()
    assert str_model.islower()
    assert not str_model.isupper()
    assert not str_model.istitle()
