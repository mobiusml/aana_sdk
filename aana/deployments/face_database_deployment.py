from pathlib import Path
from typing import Any

import faiss
import numpy as np
from pydantic import BaseModel
from ray import serve
from huggingface_hub import hf_hub_download
import tarfile

from aana.deployments.base_deployment import BaseDeployment, test_cache


class FaceDatabaseConfig(BaseModel):
    """The configuration for the face database deployment.

    Attributes:
        face_threshold (float): max distance between query face feature and reference face feature to be considered a match. (Do not go above 1.20)
        facenorm_threshold (float): min norm of query face feature and reference face feature to be considered a match. (Do not go below 16.0)
        face_features_directory (Path): Path to where the face features will be stored
        feature_extractor_name (str): Name of the face feature extractor model. This name will be used as folder name in face_features_directory to store the features in.
        hugging_face_token (str): Hugging Face token to access face database
    """

    face_threshold: float
    facenorm_threshold: float
    face_features_directory: Path
    feature_extractor_name: str
    hugging_face_token: str
    face_group_name: str = "default"


@serve.deployment  # Comment
class FaceDatabaseDeployment(BaseDeployment):
    """Deployment to serve face detector."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        """
        config_obj = FaceDatabaseConfig(**config)

        self.feat_dim = 512

        self.face_threshold = config_obj.face_threshold
        self.facenorm_threshold = config_obj.facenorm_threshold
        self.facefeat_directory = (
            Path(config_obj.face_features_directory) / config_obj.feature_extractor_name
        )

        if not Path.exists(self.facefeat_directory):
            Path.mkdir(self.facefeat_directory, parents=True)
            try:
                path_to_tarfile = hf_hub_download(
                    repo_id="mobiuslabsgmbh/aana_facedb",
                    repo_type="dataset",
                    filename="AdaFace/{}32K_{}.tar".format(
                        config_obj.face_group_name, config_obj.feature_extractor_name
                    ),
                    local_dir=self.facefeat_directory,
                    token=config_obj.hugging_face_token,
                )
                # Open the tar file and extract its contents
                with tarfile.open(path_to_tarfile, "r:*") as tar:
                    tar.extractall(path=self.facefeat_directory)
            except:
                print(
                    "Could not download face features from hugging face hub. Initializing empty database."
                )

        feature_files = list(self.facefeat_directory.glob("*.npy"))
        num_feats = len(feature_files)

        # load features and build index
        self.person_ids = []
        self.image_ids = []
        self.features = []

        for feature_file in feature_files:
            feat = np.load(feature_file).astype("float32")[
                0
            ]  # Remove singleton dimension

            feat_name = Path(
                feature_file
            ).stem  # Extract filename without file extension
            person_id, image_id = feat_name.split("|||")
            self.person_ids.append(person_id)
            self.image_ids.append(image_id)
            self.features.append(feat)

        if num_feats > 0:
            self.features = np.array(self.features).astype("float32")
            self.index = faiss.IndexFlatL2(self.feat_dim)
            self.index.add(self.features)

        else:
            self.features = np.empty((0, self.feat_dim), dtype="float32")

    @test_cache
    async def add_reference_face(
        self, face_feature: list, face_norm: float, person_name: str, image_id: str
    ) -> str:
        """Add face to reference database.

        Args:
            face_feature (np.array): face feature for person to be added
            person_name (str): name of the person to be added
            image_id (str): image id of the image used

        Returns:
            str: success
        """
        if image_id not in self.image_ids:
            if face_norm < self.facenorm_threshold:
                return "facenorm_too_low"
            else:
                self.person_ids.append(person_name)
                self.image_ids.append(image_id)

                face_feature = np.expand_dims(face_feature, axis=0).astype("float32")

                if len(self.features) == 0:
                    self.index = faiss.IndexFlatL2(self.feat_dim)

                self.index.add(face_feature)

                self.features = np.append(self.features, face_feature, axis=0)

                np.save(
                    Path(self.facefeat_directory) / f"{person_name}|||{image_id}.npy",
                    face_feature,
                )

                return "success"
        else:
            return "already_exists"

    @test_cache
    async def get_all_identities(self) -> list[str]:
        """Get all person names in database.

        Args:
            None

        Returns:
            list[str]: The names of all people in the database
        """
        return self.person_ids

    # @test_cache
    # async def search(self, face_features: list) -> dict:
    #     """Extract face features for all faces. This is the main method for identifying faces.

    #     Args:
    #         face_features (list[np.array]): list of face_features

    #     Returns:
    #         dict: dict with matched identities.
    #     """
    #     query_features = np.array(face_features)
    #     distances, indices = self.index.search(query_features, 1)
    #     results = []
    #     for i, (distances_i, indices_i) in enumerate(
    #         zip(distances, indices, strict=False)
    #     ):
    #         distance = distances_i[0]
    #         index = indices_i[0]

    #         if index == -1:
    #             results.append(
    #                 {"person_id": "unknown", "image_id": "unknown", "distance": 0.0}
    #             )
    #         elif distance > self.face_threshold:
    #             results.append(
    #                 {
    #                     "person_id": "unknown",
    #                     "image_id": "unknown",
    #                     "distance": float(distance),
    #                 }
    #             )
    #         else:
    #             results.append(
    #                 {
    #                     "person_id": self.person_ids[index],
    #                     "image_id": self.image_ids[index],
    #                     "distance": float(distance),
    #                 }
    #             )
    #     return {
    #         "identities": results,
    #    }

    # facefeat_output["facefeats_per_image"][0]["face_feats"]
    @test_cache
    async def identify_faces(self, face_features_per_image: list[dict]) -> dict:
        """Extract face features for all faces in multiple images. This is the main method for identifying faces.

        Args
            face_features_per_image: list of dicts with face features and norms per image

        Returns:
            dict: dict with matched identities, per image.
        """
        results_per_image = []
        for face_features in face_features_per_image:
            if len(face_features["face_feats"]) > 0:
                query_features = np.array(face_features["face_feats"])
                distances, indices = self.index.search(query_features, 1)
                results = []
                for i, (distances_i, indices_i) in enumerate(
                    zip(distances, indices, strict=False)
                ):
                    distance = distances_i[0]
                    index = indices_i[0]
                    face_norm = face_features["norms"][i][0]

                    if index == -1:
                        results.append(
                            {
                                "person_name": "unknown",
                                "image_id": "unknown",
                                "distance": 0.0,
                            }
                        )
                    elif distance > self.face_threshold:
                        results.append(
                            {
                                "person_name": "unknown",
                                "image_id": "unknown",
                                "distance": float(distance),
                                "norm": face_norm,
                                "quality": "bad",
                            }
                        )
                    else:
                        if face_norm >= self.facenorm_threshold:
                            results.append(
                                {
                                    "person_name": self.person_ids[index],
                                    "image_id": self.image_ids[index],
                                    "distance": float(distance),
                                    "norm": face_norm,
                                    "quality": "good",
                                }
                            )
                        else:
                            results.append(
                                {
                                    "person_name": self.person_ids[index],
                                    "image_id": self.image_ids[index],
                                    "distance": float(distance),
                                    "norm": face_norm,
                                    "quality": "bad",
                                }
                            )

                results_per_image.append(results)
            else:
                results_per_image.append("No faces identified")

        return {
            "identities_per_image": results_per_image,
        }