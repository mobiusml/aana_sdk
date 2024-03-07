# ruff: noqa: S101
import pytest

from aana.deployments.standard_concepts2_deployment import (
    filter_tags,
    map_predictions,
    remove_antonyms,
)


@pytest.fixture
def stop_list():
    """Returns a stop list."""
    return {"foo", "bar"}


def test_mapping():
    """Tests that map_predictions maps tags correctly."""
    # Arrange
    mappings_dict = {"foo": "bar", "dog": "shaggy_dog"}

    tags = ["duck", "dog", "foo", "bar"]
    scores = [1.0, 0.75, 0.5, 0.4]

    # Act
    result = map_predictions(tags, scores, mappings_dict)

    # Assert
    assert result[0][0] == "duck"
    assert result[1][0] == "shaggy_dog"
    assert result[2][0] == "bar"
    assert result[2][1] == 0.5
    assert len(result) == 3


@pytest.mark.parametrize(
    "tags, expected_tags",
    [
        (
            [("duck", 1.0), ("dog", 0.75), ("foo", 0.5), ("bar", 0.4)],
            {"duck", "dog", "foo"},
        ),
        (
            [("duck", 1.0), ("dog", 0.75), ("bar", 0.5), ("foo", 0.4)],
            {"duck", "dog", "bar"},
        ),
    ],
)
def test_remove_antonyms(tags, expected_tags):
    """Tests that removing antonyms works correctly."""
    # Arrange
    antonyms = {"foo": {"bar"}, "bar": {"foo"}, "lots of people": {"no people"}}

    # Act
    results = remove_antonyms(tags, antonyms)

    # Assert
    assert {x[0] for x in results} == expected_tags


@pytest.mark.parametrize(
    "tags, expected_tags",
    [
        (
            [("duck", 1.0), ("dog", 0.75), ("foo", 0.5), ("bar", 0.4)],
            {"duck", "dog"},
        ),
        (
            [("duck", 1.0), ("foo", 0.75), ("bar", 0.5), ("dog", 0.4)],
            {"duck"},
        ),
    ],
)
def test_filter_tags(tags, expected_tags, stop_list):
    """Tests that tag filtering works."""
    # Arrange
    confidence_threshold = 0.55

    # Act
    result = filter_tags(tags, confidence_threshold, stop_list)

    # Assert
    assert {x[0] for x in result} == expected_tags
