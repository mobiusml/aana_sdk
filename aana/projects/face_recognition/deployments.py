from aana.configs.settings import settings
from aana.deployments.face_database_deployment import (
    FaceDatabaseConfig,
    FaceDatabaseDeployment,
)
from aana.deployments.face_detection_deployment import (
    FaceDetectorConfig,
    FaceDetectorDeployment,
)
from aana.deployments.face_featureextraction_deployment import (
    FacefeatureExtractorConfig,
    FacefeatureExtractorDeployment,
)

FACEFEATURE_MODEL = "ir_101_webface4M"

deployments = {}

face_detector_deployment = FaceDetectorDeployment.options(
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.3},
    user_config=FaceDetectorConfig(
        nms_thresh=0.4,
        batch_size=4,
        input_size=640,
    ).model_dump(mode="json"),
)

deployments["face_detector_deployment"] = face_detector_deployment


facefeat_extractor_deployment = FacefeatureExtractorDeployment.options(
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.2},
    user_config=FacefeatureExtractorConfig(
        feature_extractor_name=FACEFEATURE_MODEL,
        min_face_norm=19.0,
    ).model_dump(mode="json"),
)

deployments["facefeat_extractor_deployment"] = facefeat_extractor_deployment


facedatabase_deployment = FaceDatabaseDeployment.options(
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.0},
    user_config=FaceDatabaseConfig(
        face_threshold=1.18,
        facenorm_threshold=19.0,
        face_features_directory=settings.artifacts_dir / "face_features_database",
        feature_extractor_name=FACEFEATURE_MODEL,
    ).model_dump(mode="json"),
)

deployments["facefeat_extractor_deployment"] = facefeat_extractor_deployment
