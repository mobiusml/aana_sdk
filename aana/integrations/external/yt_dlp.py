import hashlib
from pathlib import Path

import yt_dlp
from yt_dlp.utils import DownloadError

from aana.configs.settings import settings
from aana.core.models.video import Video, VideoInput
from aana.exceptions.io import (
    DownloadException,
)


def download_video(video_input: VideoInput | Video) -> Video:
    """Downloads videos for a VideoInput object.

    Args:
        video_input (VideoInput): the video input to download

    Returns:
        Video: the video object
    """
    if isinstance(video_input, Video):
        return video_input
    if video_input.url is not None:
        video_dir = settings.video_dir
        url_hash = hashlib.md5(
            video_input.url.encode(), usedforsecurity=False
        ).hexdigest()

        # we use yt_dlp to download the video
        # it works not only for youtube videos, but also for other websites and direct links
        ydl_options = {
            "outtmpl": f"{video_dir}/{url_hash}.%(ext)s",
            "extract_flat": True,
            "hls_prefer_native": True,
            "extractor_args": {"youtube": {"skip": ["hls", "dash"]}},
        }
        try:
            with yt_dlp.YoutubeDL(ydl_options) as ydl:
                info = ydl.extract_info(video_input.url, download=False)
                title = info.get("title", "")
                description = info.get("description", "")
                path = Path(ydl.prepare_filename(info))
                if not path.exists():
                    ydl.download([video_input.url])
                if not path.exists():
                    raise DownloadException(video_input.url)
                return Video(
                    path=path,
                    url=video_input.url,
                    media_id=video_input.media_id,
                    title=title,
                    description=description,
                )
        except DownloadError as e:
            # removes the yt-dlp request to file an issue
            error_message = e.msg.split(";")[0]
            raise DownloadException(url=video_input.url, msg=error_message) from e
    else:
        return video_input.convert_input_to_object()
