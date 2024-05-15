# ruff: noqa: S101

from aana.api.models.question import Question


def test_question_creation():
    """Test that a question can be created."""
    question = Question("What is the capital of France?")
    assert question == "What is the capital of France?"
