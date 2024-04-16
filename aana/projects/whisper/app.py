from aana.configs.deployments import vad_deployment, whisper_medium_deployment
from aana.projects.whisper.endpoints import (
    delete_media_endpoint,
    load_transcription_endpoint,
    transcribe_video_endpoint,
    transcribe_video_in_chunks_endpoint,
)
from aana.sdk import AanaSDK

aana_sdk = AanaSDK(port=8000)

aana_sdk.register_deployment(
    "asr_deployment",
    whisper_medium_deployment,
)

aana_sdk.register_deployment(
    "vad_deployment",
    vad_deployment,
)

aana_sdk.register_endpoint(
    name="whisper_transcribe",
    path="/video/transcribe",
    summary="Transcribe a video",
    func=transcribe_video_endpoint,
)

aana_sdk.register_endpoint(
    name="whisper_transcribe_in_chunks",
    path="/video/transcribe_in_chunks",
    summary="Transcribe a video using Whisper by segmenting it into chunks",
    func=transcribe_video_in_chunks_endpoint,
)

aana_sdk.register_endpoint(
    name="load_transcription",
    path="/video/get_transcription",
    summary="Load a transcription",
    func=load_transcription_endpoint,
)

aana_sdk.register_endpoint(
    name="delete_media",
    path="/video/delete",
    summary="Delete a media",
    func=delete_media_endpoint,
)

aana_sdk.deploy(blocking=True)
