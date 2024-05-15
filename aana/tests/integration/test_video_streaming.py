# # ruff: noqa: S101
# # Test chat with video endpoints.

# import pytest

# from aana.tests.utils import is_gpu_available, is_using_deployment_cache

# TARGET = "video_streaming"

# STREAM_INDEX_ENDPOINT = "/video/index_stream"


# @pytest.mark.skipif(
#     not is_gpu_available() and not is_using_deployment_cache(),
#     reason="GPU is not available",
# )
# @pytest.mark.parametrize("call_endpoint", [TARGET], indirect=True)
# @pytest.mark.parametrize(
#     "stream_type, stream",
#     [
#         (
#             "SRT",
#             {
#                 "url": "srt://localhost:7000",
#                 "media_id": "0",
#             },
#         ),
#         (
#             "SRT",
#             {
#                 "url": "https://localhost:7777/video.m3u8",
#                 "media_id": "1",
#             },
#         ),
#     ],
# )
# def caption_stream(streamer_setup, call_endpoint, stream_type, stream):
#     """Test chat with video endpoint."""
#     streamer_setup(stream_type, 7777, "aana/tests/files/videos/squirrel.mp4")
#     call_endpoint(
#         STREAM_INDEX_ENDPOINT,
#         {"input_stream": stream},
#     )
