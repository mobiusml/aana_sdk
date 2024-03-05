from collections import defaultdict
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import torch
import yaml
from pydantic import BaseModel, Field
from ray import serve
from typing_extensions import TypedDict

from aana.deployments.base_deployment import BaseDeployment
from aana.exceptions.general import ImageReadingException, InferenceException
from aana.models.core.image import Image
from aana.utils.test import test_cache  # noqa: F401


class StandardConceptsV2Config(BaseModel):
    """The configuration for the Standrd Concepts V2 deployment.

    Attributes:
        path_model (str): string path to the model file
        path_keywords (str): string path to the list of keywords
        encoding_keywords (str): encoding of strings in the keywords file
        encoding_config (str): encoding of the config file
        path_config (str): sttringpath to the config yaml
        image_size (int | tuple[int,int]): size to resize the model
        confidence_threshold (float): confidence threshold to apply
    """

    path_model: str
    path_keywords: str
    encoding_keywords: str
    path_config: str
    encoding_config: str
    image_size: int | tuple[int] | list[int]
    confidence_threshold: float = Field(default=0.55)
    top_n: int = Field(default=20)


class KeywordOutput(TypedDict):
    """Output for a single keyword.

    Attributes:
        name (str): name of tag or keyword.
        score (float): the confidence score the model gives for the tag.
    """

    name: str
    score: float


class CategoryOutput(TypedDict):
    """Output for a category.

    Attributes:
        name (str): name of the category.
        keywords (list[KeywordOutputt]): list of keyword results.
    """

    name: str
    items: list[KeywordOutput]


CategorizedTaggingOutput: TypeAlias = list[CategoryOutput]
"""Output for tagging with categories."""


UncategorizedTaggingOutput: TypeAlias = list[KeywordOutput]
"""Output type for tagging if categories not included."""

TaggingPredictions: TypeAlias = UncategorizedTaggingOutput | CategorizedTaggingOutput
"""Tag predictions. Either categorized or uncategorized."""


class TaggingOutput(TypedDict):
    """Output type for tagging. Contains features and either a list of KeywordOutputs, or a list of CategoryOutputs.

    Attributes:
        features (np.ndarray): Feature vectors from the model
        prediction: (TaggingPredictions): Either categorized or uncategorized output
    """

    features: np.ndarray
    predictions: TaggingPredictions


@serve.deployment
class StandardConceptsV2Deployment(BaseDeployment):
    """Deployment to serve StandardConcepts v2 models using local file."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        It loads the model and processor from HuggingFace.

        The configuration should conform to the HFBlip2Config schema.
        """
        self._config = StandardConceptsV2Config(**config)

        # Load model
        with Path(self._config.path_model).open("rb") as f:
            self._model = torch.jit.load(f)
            if torch.cuda.is_available():
                self._use_gpu = True
                self._model = self._model.cuda()

        with Path(self._config.path_keywords).open("rb") as f:
            concepts_list = np.load(f)
            self._concepts = np.array(
                x.decode(self._config.encoding_keywords) for x in concepts_list
            )

        with Path(self._config.path_config).open(
            encoding=self._config.encoding_config
        ) as f:
            concepts = yaml.safe_load(f, Loader=yaml.CBaseLoader)
            stop_list = {x[1] for x in concepts.get("stop_list", {}).items()}
            mappings = {
                a: b
                for mapping in concepts["mappings"].values()
                for a, b in mapping.items()
            }
            antonyms: defaultdict[str, set] = defaultdict(set)
            for a, b in concepts.get("antonyms", {}):
                antonyms[a] += b
                antonyms[b] += a
            categories = concepts.get("categories", {})
            thresholds = concepts.get("thresholds", {})
            self._concept_config = {
                "antonyms": antonyms,
                "categories": categories,
                "mappings": mappings,
                "stop_list": stop_list,
                "thresholds": thresholds,
            }

    def generate(self, image: Image) -> TaggingOutput:  # noqa: C901
        """Generate captions for the given images.

        This method is called by the batch processor.

        Args:
            image (Image): the image to process.

        Returns:
            TaggingOutput: the features and predictions for the image

        Raises:
            InferenceException: if the inference fails
        """
        # Check that input is the right size
        data = image.get_numpy()
        if not data.shape == self._config.image_size:
            raise ImageReadingException(image)

        # Inference
        try:
            batch = torch.from_numpy([data])
            if self._use_gpu:
                batch = batch.cuda()
            result = self._model(batch)
            pred, features = tuple(x.cpu().detach().numpy() for x in result)
        except Exception as e:
            raise InferenceException("standard_concepts_v2") from e

        # Postprocessing
        # Get indices of preds exceeding confidence threshold, sorted by score
        indices = sorted(
            np.where(pred > self._config.confidence_threshold)[0],
            key=lambda i: -pred[i],
        )
        scores = pred[indices].tolist()
        tags = self._concepts[indices].tolist()

        mapped_predictions = list[tuple[str, float]]()
        seen_tags = set()
        for tag, score in zip(tags, scores, strict=True):
            # map the tag to the value in the config
            tag = self._concept_config["mappings"].get(tag, tag)
            # if already seen, don't add again
            if tag in seen_tags:
                continue
            mapped_predictions.append((tag, score))
            seen_tags.add(tag)
        # filter tags
        filtered_tags = list[tuple[str, float]]()
        for tag, score in mapped_predictions:
            # if score below threshold, don't add
            if score < self._config.confidence_threshold:
                continue
            if tag in self._concept_config["stop_list"]:
                continue
            filtered_tags.append((tag, score))

        # Remove antonyms
        cleaned_tags = list[tuple[str, float]]()
        seen_tags: set = set()
        for tag, score in filtered_tags:
            # Add if we have not seen an antonym to this tag already
            # (~~Un~~Cool kids say, "Add if the set of seen tags and
            # the set of antonym tags is the empty set")
            if not seen_tags & self._concept_config["antonyms"].get(tag, set()):
                seen_tags.add(tag)
                cleaned_tags.append((tag, score))

        # Having cleaned tags, we now take into account top_n
        processed_tags = cleaned_tags[: self._config.top_n]

        # Having cleaned tags and limited ourselves to n items,
        # we now structure output
        # TODO: allow uncategorized output
        categories = defaultdict(list)
        uncategorized = []
        for tag, score in processed_tags:
            # Get categories for each tag
            # (one tag can have multiple categories, apparently)
            tag_categories = self._concept_config["categories"].get(tag, [])
            if not tag_categories:
                uncategorized.append({"name": tag, "score": score})
                continue
            for category in tag_categories:
                categories[category].append({"name": tag, "score": score})
        final_categories = [
            {"category": category, "items": items}
            for category, items in categories.items()
        ]
        if uncategorized:
            final_categories.append(
                {
                    "category": "uncategorized",
                    "items": uncategorized,
                }
            )
        return {"features": features, "predictions": final_categories}
