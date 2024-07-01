from typing import Any

import numpy as np
from pydantic import BaseModel
from ray import serve

from aana.core.models.image import Image
from aana.deployments.base_deployment import BaseDeployment, test_cache
from aana.processors.facefeat_extractor import (
    face_align_landmarks,
    load_IR50_model,
    to_input,
)


class FacefeatureExtractorConfig(BaseModel):
    """The configuration for the face detector deployment.

    Attributes:
        model_url (str): the model URL
    """

    model_url: str
    min_face_norm: float


@serve.deployment
class FacefeatureExtractorDeployment(BaseDeployment):
    """Deployment to serve face detector."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        """
        config_obj = FacefeatureExtractorConfig(**config)
        self.model_url = config_obj.model_url
        self.min_face_norm = config_obj.min_face_norm

        # Load the model
        self.facefeat_extractor, self.device = load_IR50_model(self.model_url)

    @test_cache
    async def predict(self, images: list[Image], face_landmarks: list[list]) -> dict:
        """Extract face features for all faces.

        Args:
            images (list[Image]): the images
            face_landmarks (list[list]): per image, it contains a list of the face_landmarks for all faces that features will be extracted.

        Returns:
            dict: the predictions
        """
        facefeats_per_image = []

        for k, image in enumerate(images):
            image_np = image.get_numpy()

            face_crops = face_align_landmarks(
                image_np, face_landmarks[k], image_size=(112, 112), method="similar"
            )

            prep_crops = to_input(face_crops, self.device)

            face_feats, norms = self.facefeat_extractor(prep_crops)
            face_feats = face_feats.detach().cpu().numpy()

            norms = norms.detach().cpu().numpy()

            # Filter out faces with low norms (i.e., low quality, see AdaFace paper for reasoning)
            # Filter out faces with low norms (i.e., low quality, see AdaFace paper for reasoning)
            face_ids_above_minnorm = np.where(norms >= self.min_face_norm)[0]

            facefeats_per_image.append(
                {
                    "face_feats": face_feats[face_ids_above_minnorm],
                    "norms": norms[face_ids_above_minnorm],
                }
            )

        return {"facefeats_per_image": facefeats_per_image}
