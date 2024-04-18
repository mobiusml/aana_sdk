from aana.configs.deployments import vad_deployment, whisper_medium_deployment
from aana.projects.whisper.endpoints import (
    DeleteMediaEndpoint,
    LoadTranscriptionEndpoint,
    TranscribeVideoEndpoint,
    TranscribeVideoInChunksEndpoint,
)
from aana.sdk import AanaSDK

aana_app = AanaSDK(port=8000)

aana_app.register_deployment(
    "asr_deployment",
    whisper_medium_deployment,
)

aana_app.register_deployment(
    "vad_deployment",
    vad_deployment,
)

aana_app.register_endpoint(
    name="whisper_transcribe",
    path="/video/transcribe",
    summary="Transcribe a video",
    endpoint_cls=TranscribeVideoEndpoint,
)

aana_app.register_endpoint(
    name="whisper_transcribe_in_chunks",
    path="/video/transcribe_in_chunks",
    summary="Transcribe a video using Whisper by segmenting it into chunks",
    endpoint_cls=TranscribeVideoInChunksEndpoint,
)

aana_app.register_endpoint(
    name="load_transcription",
    path="/video/get_transcription",
    summary="Load a transcription",
    endpoint_cls=LoadTranscriptionEndpoint,
)

aana_app.register_endpoint(
    name="delete_media",
    path="/video/delete",
    summary="Delete a media",
    endpoint_cls=DeleteMediaEndpoint,
)

if __name__ == "__main__":
    aana_app.deploy(blocking=True)
