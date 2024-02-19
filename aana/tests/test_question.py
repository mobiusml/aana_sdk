# ruff: noqa: S101
import pytest

from aana.models.pydantic.question import Question


def test_question_creation():
    """Test that a question can be created."""
    question = Question(__root__="What is the capital of France?")
    assert question == "What is the capital of France?"

    question = Question("What is the capital of France?")
    assert question == "What is the capital of France?"

    with pytest.raises(ValueError):
        question = Question()
