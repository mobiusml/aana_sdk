# ruff: noqa: ASYNC101
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
    """Deployment to serve StandardConcepts v2 models using local file.

    For a production deployment, we wouldn't implement it this way. We would have
    at least three separate deployments: one for resizing, another for inference,
    and a third for post-processing; with a shared backbone architecture like we
    had for CLIP keywording, there would even be two inferences, one for feature
    extraction and a second one for actual tag predictions. Thihs allows you to
    optimally split up parts of the workflow to optimally use CPU, GPU, and multi-
    server resources.

    These all should be in deployments because they require configuration to run,
    which is stateful, and hence unsuitable for a task or function.

    Actually, in an ideal world the concept configurations would live in a database,
    so it could be easily modified with other endpoints. Adding to the stop list
    or changing a mapping could be as easy as API calls, and adding a new language
    could be a few (okay: a lot) of API calls. The backbone model would need to be
    precomposed but you could even put the concept weights and biases in the (TBC)
    vector datastore and build that whole layer up at runtime.

    We'd also pass a parameters input object instead of requiring a single value for
    thresholding and `top_n`.
    """

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        It loads the model and processor from the filesystem.

        The configuration should conform to the StandardConceptsV2Config schema.
        """
        self._config = StandardConceptsV2Config(**config)

        # Load model
        with Path(self._config.path_model).open("rb") as f:
            self._model = torch.jit.load(f)
            if torch.cuda.is_available():
                self._use_gpu = True
                self._model = self._model.cuda()
            else:
                self._use_gpu = False

        with Path(self._config.path_keywords).open("rb") as f:
            concepts_list = np.load(f)
            self._concepts = np.array(
                [x.decode(self._config.encoding_keywords) for x in concepts_list]
            )

        with Path(self._config.path_config).open(
            encoding=self._config.encoding_config
        ) as f:
            concepts = yaml.safe_load(f)

            stop_list = {x for xs in concepts.get("stop_list", {}).values() for x in xs}
            mappings = {
                a: b
                for mapping in concepts["mappings"].values()
                for a, b in mapping.items()
            }
            antonyms: defaultdict[str, set] = defaultdict(set)
            for a, b in concepts.get("antonyms", {}):
                antonyms[a].add(b)
                antonyms[b].add(a)
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
        """Generate tags for the given image.

        Args:
            image (Image): the image to process.

        Returns:
            TaggingOutput: the features and predictions for the image

        Raises:
            ImageReadingException: if the miage is not the right size
            InferenceException: if the inference fails
        """
        # Check that input is the right size
        data = image.get_numpy()
        if data.shape != tuple(self._config.image_size):
            raise ImageReadingException(image)

        # Inference
        try:
            batch = torch.from_numpy(np.stack((data, np.zeros_like(data))))

            if self._use_gpu:
                batch = batch.cuda()
            result = self._model(batch)
            pred, features = tuple(x.cpu().detach().numpy() for x in result)
        except Exception as e:
            print(e)
            raise InferenceException("standard_concepts_v2") from e

        # Postprocessing
        # Get indices of preds exceeding confidence threshold, sorted by score
        pred = pred[0]  # discard zeros output
        indices = sorted(
            np.where(pred > self._config.confidence_threshold)[0],
            key=lambda i: -pred[i],
        )
        scores = pred[indices].tolist()
        tags = self._concepts[indices].tolist()

        # Here we're abusing notation to use a type annotation as a constructor.
        # The created list doesn't actually enforce the generic type annnotations
        # at run time, but a typechecker can verify them before that.
        processed_tags = list[tuple[str, float]]()
        # Gets flagged by MyPy:
        # processed_tags.append(0)
        seen_tags: set = set()
        for tag, score in zip(tags, scores, strict=True):
            # map the tag to the value in the config
            tag = self._concept_config["mappings"].get(tag, tag)
            # If already seen, don't add again
            if tag in seen_tags:
                continue
            # Filter if below threshohld or on stoplist
            if score < self._config.confidence_threshold:
                continue
            if self._concept_config["stop_list"]:
                continue
            # Add if we have not seen an antonym to this tag already
            # (~~Un~~Cool kids say, "Add if the set of seen tags and
            # the set of antonym tags is the empty set")
            if not seen_tags & self._concept_config["antonyms"].get(tag, set()):
                seen_tags.add(tag)
                processed_tags.append((tag, score))
            # Only take the top n results
            if len(processed_tags) >= self._config.top_n:
                break

        # Having cleaned tags and limited ourselves to n items,
        # we now structure output
        # TODO: allow uncategorized output
        categories: defaultdict[str, list] = defaultdict(list)
        uncategorized: UncategorizedTaggingOutput = []
        for tag, score in processed_tags:
            # Get categories for each tag
            # (one tag can have multiple categories, apparently)
            tag_categories = self._concept_config["categories"].get(tag, [])
            if not tag_categories:
                uncategorized.append({"name": tag, "score": score})
                continue
            for category in tag_categories:
                categories[category].append({"name": tag, "score": score})
        final_categories: list[CategoryOutput] = [
            {"name": category, "items": items} for category, items in categories.items()
        ]
        if uncategorized:
            final_categories.append(
                {
                    "name": "uncategorized",
                    "items": uncategorized,
                }
            )
        return {"features": features, "predictions": final_categories}
