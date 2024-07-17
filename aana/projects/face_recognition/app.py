from deployments import (
    face_detector_deployment,
    facedatabase_deployment,
    facefeat_extractor_deployment,
)

from aana.projects.face_recognition.endpoints import (
    AddReferenceFaceEndpoint,
    FaceFeatureExtractionEndpoint,
    RecognizeFacesEndpoint
)
from aana.sdk import AanaSDK

deployments = [
    {
        "name": "face_detector_deployment",
        "instance": face_detector_deployment,
    },
    {
        "name": "facefeat_extractor_deployment",
        "instance": facefeat_extractor_deployment,
    },
    {
        "name": "facedatabase_deployment",
        "instance": facedatabase_deployment,
    },
]

endpoints = [
    {
        "name": "extract_face_features",
        "path": "/extract_face_features",
        "summary": "Detect faces and extract their face features",
        "endpoint_cls": FaceFeatureExtractionEndpoint,
    },
    {
        "name": "recognize_faces",
        "path": "/recognize_faces",
        "summary": "Detect and identify faces",
        "endpoint_cls": RecognizeFacesEndpoint,
    },
    # {
    {
        "name": "add_reference_face",
        "path": "/add_reference_face",
        "summary": "Add a reference face to the face database",
        "endpoint_cls": AddReferenceFaceEndpoint,
    },
]

aana_app = AanaSDK(name="face_recognition")


for deployment in deployments:
    aana_app.register_deployment(
        name=deployment["name"],
        instance=deployment["instance"],
    )

for endpoint in endpoints:
    aana_app.register_endpoint(
        name=endpoint["name"],
        path=endpoint["path"],
        summary=endpoint["summary"],
        endpoint_cls=endpoint["endpoint_cls"],
        event_handlers=endpoint.get("event_handlers", []),
    )


if __name__ == "__main__":
    aana_app.connect(
        host="127.0.0.1", port=8000, show_logs=False
    )  # Connects to the Ray cluster or starts a new one.
    # aana_app.migrate()  # Runs the migrations to create the database tables.
    aana_app.deploy(blocking=True)
