from typing import TYPE_CHECKING, Annotated, TypedDict

import numpy as np
from pydantic import Field

from aana.api.api_generation import Endpoint
from aana.core.models.image import ImageInput, ImageInputList
from aana.core.models.video import VideoInput, VideoMetadata, VideoParams
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.integrations.external.yt_dlp import download_video
from aana.processors.remote import run_remote
from aana.integrations.external.decord import generate_frames
from aana.processors.face_database_helpers import (
    parse_faceresults_images,
    parse_faceresults_video,
)

if TYPE_CHECKING:
    from aana.core.models.video import Video


class FaceFeatureExtractionEndpointOutput(TypedDict):
    """Face Recognition endpoint output."""

    # image: Annotated[
    #     list, Field(description="The generated image as a array of pixels.")
    # ]
    face_features_per_image: list
    # norms: list


class FaceFeatureExtractionEndpoint(Endpoint):
    """Face Detection endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        self.face_detector_handle = await AanaDeploymentHandle.create(
            "face_detector_deployment"
        )
        self.face_featextractor_handle = await AanaDeploymentHandle.create(
            "facefeat_extractor_deployment"
        )

    async def run(self, images: ImageInputList) -> FaceFeatureExtractionEndpointOutput:
        """Run the face detection endpoint."""
        images = images.convert_input_to_object()
        face_detection_output = await self.face_detector_handle.predict(images)

        face_featextract_output = await self.face_featextractor_handle.predict(
            images, face_detection_output["keypoints"]
        )

        return FaceFeatureExtractionEndpointOutput(
            face_features_per_image=face_featextract_output["facefeats_per_image"]
        )


class AddReferenceFaceEndpointOutput(TypedDict):
    """Add reference face endpoint output."""

    status: str


class AddReferenceFaceEndpoint(Endpoint):
    """Add a face to the reference face database endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        self.face_detector_handle = await AanaDeploymentHandle.create(
            "face_detector_deployment"
        )
        self.face_featextractor_handle = await AanaDeploymentHandle.create(
            "facefeat_extractor_deployment"
        )
        self.face_database_handle = await AanaDeploymentHandle.create(
            "facedatabase_deployment"
        )

    async def run(
        self, image: ImageInput, person_name: str, image_id: str
    ) -> AddReferenceFaceEndpointOutput:
        """Add a reference face to the face database."""

        image = image.convert_input_to_object()
        face_detection_output = await self.face_detector_handle.predict([image])

        face_featextract_output = await self.face_featextractor_handle.predict(
            [image], face_detection_output["keypoints"]
        )

        face_featextract_output = face_featextract_output["facefeats_per_image"][0]
        if (
            len(face_featextract_output["face_feats"]) == 1
        ):  # Only one face detected. Add to database
            face_feat = face_featextract_output["face_feats"][0]
            # face_norm = face_featextract_output["norms"][0]

            add_face_status = await self.face_database_handle.add_reference_face(
                face_feat, person_name, image_id
            )

            return AddReferenceFaceEndpointOutput(status=add_face_status)

        else:
            return AddReferenceFaceEndpointOutput(status="failed")


class RecognizeFacesEndpointOutput(TypedDict):
    """Face recognition endpoint output."""

    identified_faces_per_image: dict


class RecognizeFacesEndpoint(Endpoint):
    """Face recognition endpoint endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        self.face_detector_handle = await AanaDeploymentHandle.create(
            "face_detector_deployment"
        )
        self.face_featextractor_handle = await AanaDeploymentHandle.create(
            "facefeat_extractor_deployment"
        )
        self.face_database_handle = await AanaDeploymentHandle.create(
            "facedatabase_deployment"
        )

    async def run(self, images: ImageInputList) -> RecognizeFacesEndpointOutput:
        """Recognize faces in images endpoint."""
        images = images.convert_input_to_object()
        face_detection_output = await self.face_detector_handle.predict(images)

        face_featextract_output = await self.face_featextractor_handle.predict(
            images, face_detection_output["keypoints"]
        )

        identified_faces = await self.face_database_handle.identify_faces(
            face_featextract_output["facefeats_per_image"]
        )

        bboxes_per_frame = face_detection_output["bounding_boxes"]
        faceid_output = parse_faceresults_images(
            identified_faces["identities_per_image"], bboxes_per_frame
        )

        return RecognizeFacesEndpointOutput(identified_faces_per_image=faceid_output)


class RecognizeFacesVideoEndpointOutput(TypedDict):
    """Face recognition endpoint output."""

    recognized_persons: dict
    vide_duration: float
    # decoded_frames: list


class RecognizeFacesVideoEndpoint(Endpoint):
    """Recognize faces in video endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        self.face_detector_handle = await AanaDeploymentHandle.create(
            "face_detector_deployment"
        )
        self.face_featextractor_handle = await AanaDeploymentHandle.create(
            "facefeat_extractor_deployment"
        )
        self.face_database_handle = await AanaDeploymentHandle.create(
            "facedatabase_deployment"
        )

    async def run(
        self,
        video: VideoInput,
        video_params: VideoParams,
        # return_frames: bool = False,
    ) -> RecognizeFacesVideoEndpointOutput:
        """Transcribe video in chunks."""
        video_obj: Video = await run_remote(download_video)(video_input=video)

        detected_faces_per_frame = {
            "timestamps": [],
            "frame_ids": [],
            "identified_faces_per_frame": [],
            "bboxes_per_frame": [],
        }
        # decoded_frames = []

        video_duration = 0.0
        async for frames_dict in run_remote(generate_frames)(
            video=video_obj, params=video_params
        ):
            detected_faces_per_frame["timestamps"].extend(frames_dict["timestamps"])
            detected_faces_per_frame["frame_ids"].extend(frames_dict["frame_ids"])
            video_duration = frames_dict["duration"]

            face_detection_output = await self.face_detector_handle.predict(
                frames_dict["frames"]
            )

            face_featextract_output = await self.face_featextractor_handle.predict(
                frames_dict["frames"], face_detection_output["keypoints"]
            )

            identified_faces = await self.face_database_handle.identify_faces(
                face_featextract_output["facefeats_per_image"]
            )

            detected_faces_per_frame["bboxes_per_frame"].extend(
                face_detection_output["bounding_boxes"]
            )

            detected_faces_per_frame["identified_faces_per_frame"].extend(
                identified_faces["identities_per_image"]
            )
            # if return_frames:
            #     decoded_frames.extend(
            #         [np.array(frame).tolist() for frame in frames_dict["frames"]]
            #     )

        per_person_dict = parse_faceresults_video(detected_faces_per_frame)

        return RecognizeFacesVideoEndpointOutput(
            recognized_persons=per_person_dict,
            video_duration=video_duration,
            # decoded_frames=decoded_frames,
        )
