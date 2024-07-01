import os
import uuid
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from pydantic import BaseModel
from ray import serve

from aana.configs.settings import settings
from aana.core.models.image import Image
from aana.deployments.base_deployment import BaseDeployment, test_cache
from aana.processors.facefeat_extractor import (
    face_align_landmarks,
    load_IR50_model,
    to_input,
)


class FaceDatabaseConfig(BaseModel):
    """The configuration for the face database deployment.

    Attributes:
        face_threshold (float): max distance between query face feature and reference face feature to be considered a match. (Do not go above 1.20)
    """

    face_threshold: float
    faces_dict_file: str


@serve.deployment
class FaceDatabaseDeployment(BaseDeployment):
    """Deployment to serve face detector."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        """
        config_obj = FaceDatabaseConfig(**config)

        self.face_threshold = config_obj.face_threshold

        reference_face_dict_file = Path(
            settings.artifacts_dir / config_obj.faces_dict_file
        )

        if reference_face_dict_file.exists():
            self.ref_faces_dict = np.load(reference_face_dict_file)
        else:
            self.ref_faces_dict = {}

        # load features and build index
        self.image_ids = list(self.ref_faces_dict.keys())
        self.person_ids = [
            self.ref_faces_dict[id_]["person_id"] for id_ in self.ref_faces_dict
        ]
        self.features = [
            self.ref_faces_dict[id_]["face_feature"] for id_ in self.ref_faces_dict
        ]

        self.features = np.array(self.features)
        self.index = faiss.IndexFlatL2(self.features.shape[1])
        self.index.add(self.features)

    @test_cache
    async def add_reference_face(
        self, face_feature: list, person_name: str, image_id: str
    ) -> str:
        """Add face to reference database.

        Args:
            face_feature np.array: face feature for person to be added

        Returns:
            str: success
        """
        if image_id is None:
            image_id = uuid.uuid4()

        if image_id not in self.ref_faces_dict:
            self.ref_faces_dict[image_id] = {
                "face_feature": face_feature,
                "person_name": person_name,
            }
            self.index.add(face_feature)

            return "success"
        else:
            return "failed: ID exists"

    @test_cache
    async def get_all_identities(self) -> list[list[str]]:
        """Get all person names in database.

        Args:
            None

        Returns:
            list[str]:
        """
        all_persons = [
            [id_, self.ref_faces_dict[id_]["person_name"]]
            for id_ in self.ref_faces_dict
        ]

        return all_persons

    @test_cache
    async def search(self, face_features: list) -> dict:
        """Extract face features for all faces.

        Args:
            face_features (list[np.array]): list of face_features

        Returns:
            dict: dict with matched identities.
        """
        query_features = np.array(face_features)
        distances, indices = self.index.search(query_features, 1)
        results = []
        for i, (distances_i, indices_i) in enumerate(
            zip(distances, indices, strict=False)
        ):
            distance = distances_i[0]
            index = indices_i[0]

            if index == -1:
                results.append(
                    {"person_id": "unknown", "image_id": "unknown", "distance": 0.0}
                )
            elif distance > self.face_threshold:
                results.append(
                    {
                        "person_id": "unknown",
                        "image_id": "unknown",
                        "distance": float(distance),
                    }
                )
            else:
                results.append(
                    {
                        "person_id": self.person_ids[index],
                        "image_id": self.image_ids[index],
                        "distance": float(distance),
                    }
                )
        return {
            "identities": results,
        }
