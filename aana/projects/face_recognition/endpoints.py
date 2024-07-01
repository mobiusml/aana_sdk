from typing import Annotated, TypedDict

import numpy as np
from pydantic import Field

from aana.api.api_generation import Endpoint
from aana.core.models.image import ImageInput, ImageInputList
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle


class FaceRecognitionEndpointOutput(TypedDict):
    """Face Recognition endpoint output."""

    # image: Annotated[
    #     list, Field(description="The generated image as a array of pixels.")
    # ]
    face_features_per_image: list
    # norms: list


class FaceRecognitionEndpoint(Endpoint):
    """Face Detection endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        self.face_detector_handle = await AanaDeploymentHandle.create(
            "face_detector_deployment"
        )
        self.face_featextractor_handle = await AanaDeploymentHandle.create(
            "facefeat_extractor_deployment"
        )
        # await AanaDeploymentHandle.create("face_detector")

    async def run(self, images: ImageInputList) -> FaceRecognitionEndpointOutput:
        """Run the face detection endpoint."""

        images = images.convert_input_to_object()
        face_detection_output = await self.face_detector_handle.predict(images)

        face_featextract_output = await self.face_featextractor_handle.predict(
            images, face_detection_output["keypoints"]
        )

        return FaceRecognitionEndpointOutput(
            face_features_per_image=face_featextract_output["facefeats_per_image"]
        )


# class FaceRecognitionEndpointOutput(TypedDict):
#     """Face Recognition endpoint output."""

#     # image: Annotated[
#     #     list, Field(description="The generated image as a array of pixels.")
#     # ]
#     status: str
#     # norms: list


# class AddReferenceFaceEndpoint(Endpoint):
#     """Add a face to the reference face database endpoint."""

#     async def initialize(self):
#         """Initialize the endpoint."""
#         self.face_detector_handle = await AanaDeploymentHandle.create(
#             "face_detector_deployment"
#         )
#         self.face_featextractor_handle = await AanaDeploymentHandle.create(
#             "facefeat_extractor_deployment"
#         )

#         #Load reference face database

#     async def run(self, image: ImageInput, person_name: str, group_name: str='default') -> FaceRecognitionEndpointOutput:
#         """Run the face detection endpoint."""

#         image = image.convert_input_to_object()
#         face_detection_output = await self.face_detector_handle.predict([image])

#         face_featextract_output = await self.face_featextractor_handle.predict(
#             [image], face_detection_output["keypoints"]
#         )

#         face_featextract_output = face_featextract_output['facefeats_per_image'][0]
#         if(len(face_featextract_output['face_feats'])==1): #Only one face detected. Add to database
#             face_feat = face_featextract_output['face_feats'][0]
#             face_norm = face_featextract_output['norms'][0]
            

        

