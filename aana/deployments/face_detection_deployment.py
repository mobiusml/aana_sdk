from typing import Any

from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from ray import serve

from aana.configs.settings import settings
from aana.core.models.image import Image
from aana.deployments.base_deployment import BaseDeployment, test_cache
from aana.processors.face_detect import SCRFD, resize_image
from aana.processors.onnx_model_wrapper import ONNXModel


class FaceDetectorConfig(BaseModel):
    """The configuration for the face detector deployment.

    Attributes:
        batch_size (int): size of batch
        nms_thresh (float): threshold used for NMS (default: 0.4)
        input_size (int): input resolution for detection model (640 or 1280)
    """

    batch_size: int
    nms_thresh: float
    input_size: int


@serve.deployment
class FaceDetectorDeployment(BaseDeployment):
    """Deployment to serve face detector."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        """
        config_obj = FaceDetectorConfig(**config)
        self.batch_size = config_obj.batch_size
        self.input_size = config_obj.input_size
        if self.input_size not in [640, 1280]:
            raise ValueError("Input size for face detection must be in [640, 1280]")  # noqa: TRY003

        # Download the model weights to the local directory
        path_to_weights = hf_hub_download(
            repo_id="mobiuslabsgmbh/aana_facerecognition",
            filename="face_detection/scrfd_10g_gnkps.onnx",
            local_dir=settings.model_dir,
        )

        # Load the model
        onnx_model = ONNXModel(
            model=path_to_weights,
            batch_size=self.batch_size,
            input_size=self.input_size,
        )

        self.face_detector = SCRFD(onnx_model)
        self.face_detector.prepare(config_obj.nms_thresh)

    @test_cache
    async def predict(self, images: list[Image]) -> dict:
        """Detect faces in the images.

        Args:
            images (list[Image]): the images

        Returns:
            dict: the predictions
        """
        bboxes_per_img = []
        keypoints_per_img = []
        num_batches = len(images) // self.batch_size
        num_imgs_last_batch = len(images) - self.batch_size * num_batches

        # Process all "full" batches
        for k in range(num_batches):
            batch_start = k * self.batch_size
            batch_end = (k + 1) * self.batch_size

            image_batch = []
            resize_scales_batch = []
            for image in images[batch_start:batch_end]:
                image_np = image.get_numpy()
                image_np_resized, resize_scale = resize_image(
                    image_np, max_size=[self.input_size, self.input_size]
                )  # Important: Do not change the resize function, the model was trained using padding.
                image_batch.append(image_np_resized)
                resize_scales_batch.append(
                    resize_scale
                )  # Needed to map the bboxes and keypoints back to original resolution of the image

            # Predict batch
            bboxes_batch, keypoints_batch = self.face_detector.detect(image_batch)

            # Map the points to original image resolution
            for k in range(self.batch_size):
                bboxes_batch[k] = rescale_boxes_scale(
                    bboxes_batch[k], resize_scales_batch[k]
                )
                keypoints_batch[k] = rescale_keypoints_scale(
                    keypoints_batch[k], resize_scales_batch[k]
                )

            bboxes_per_img.extend(bboxes_batch)
            keypoints_per_img.extend(keypoints_batch)

        # Process potential last batch, which contains fewer than self.batch_size images
        if num_imgs_last_batch > 0:
            image_batch = []
            resize_scales_batch = []
            batch_start = len(images) - num_imgs_last_batch
            batch_end = len(images)

            for image in images[batch_start:batch_end]:
                image_np = image.get_numpy()
                image_np_resized, resize_scale = resize_image(
                    image_np, max_size=[self.input_size, self.input_size]
                )  # Important: Do not change the resize function, the model was trained using padding.
                image_batch.append(image_np_resized)
                resize_scales_batch.append(
                    resize_scale
                )  # Needed to map the bboxes and keypoints back to original resolution of the image

            for i in range(self.batch_size - num_imgs_last_batch):
                image_batch.append(image_np_resized)  # just add copy of last image

            # Predict batch
            bboxes_batch, keypoints_batch = self.face_detector.detect(image_batch)

            bboxes_batch = bboxes_batch[0:num_imgs_last_batch]
            keypoints_batch = keypoints_batch[0:num_imgs_last_batch]
            # Map the points to original image resolution
            for k in range(num_imgs_last_batch):
                bboxes_batch[k] = rescale_boxes_scale(
                    bboxes_batch[k], resize_scales_batch[k]
                )
                keypoints_batch[k] = rescale_keypoints_scale(
                    keypoints_batch[k], resize_scales_batch[k]
                )

            bboxes_per_img.extend(bboxes_batch)
            keypoints_per_img.extend(keypoints_batch)

        return {"bounding_boxes": bboxes_batch, "keypoints": keypoints_batch}


# Rescale box coordinates back to original resolution
def rescale_boxes_scale(boxes, scale_factor):
    for box in boxes:
        box[0] /= scale_factor
        box[2] /= scale_factor
        box[1] /= scale_factor
        box[3] /= scale_factor

    return boxes


# Rescale face landmark coordinates back to original resolution
def rescale_keypoints_scale(kpts, scale_factor):
    for kpt in kpts:
        for k in kpt:
            k[0] /= scale_factor
            k[1] /= scale_factor

    return kpts
