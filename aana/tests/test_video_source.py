from operator import not_
import pytest
from aana.models.core.video_source import VideoSource


def test_video_source_from_url():
    """
    Test that VideoSource.from_url returns the correct VideoSource for a given URL.
    """
    # Test a YouTube URL
    valid_youtube_urls = [
        "https://youtube.com/watch?v=yModCU1OVHY",
        "http://youtube.com/watch?v=yModCU1OVHY",
        "https://www.youtube.com/watch?v=yModCU1OVHY",
        "http://www.youtube.com/watch?v=yModCU1OVHY",
        "www.youtube.com/watch?v=yModCU1OVHY",
        "youtube.com/watch?v=yModCU1OVHY",
        "https://youtube.de/watch?v=yModCU1OVHY",
        "http://youtube.de/watch?v=yModCU1OVHY",
        "https://www.youtube.de/watch?v=yModCU1OVHY",
        "http://www.youtube.de/watch?v=yModCU1OVHY",
        "www.youtube.de/watch?v=yModCU1OVHY",
        "youtube.de/watch?v=yModCU1OVHY",
        "https://youtu.be/yModCU1OVHY",
        "http://youtu.be/yModCU1OVHY",
        "https://www.youtu.be/yModCU1OVHY",
        "http://www.youtu.be/yModCU1OVHY",
        "www.youtu.be/yModCU1OVHY",
        "youtu.be/yModCU1OVHY",
        "https://www.youtube.co.uk/watch?v=yModCU1OVHY",
        "https://www.youtube.co.uk/watch?v=yModCU1O",
        "https://www.youtube.com/watch?v=18pCXD709TI",
        "https://www.youtube.com/watch?v=18pCXD7",
    ]

    not_youtube_urls = [
        "https://example.com/video.mp4",
        "https://youtube/watch?v=",
        "https://www.youtubecom/watch?v=",
        "http://.youtube.com/watch?v=abc123",
        "https://youtube.co..uk/watch?v=abc123",
        "youtube/watch?v=abc123",
        "https:/youtube.com/watch?v=abc123",
        "http://youtube/watch?v=",
        "https://youtu.be/",
        "https://www.youtube.com/",
        "http://www.youtu.be/watch?v=",
    ]

    for url in valid_youtube_urls:
        assert VideoSource.from_url(url) == VideoSource.YOUTUBE

    for url in not_youtube_urls:
        assert VideoSource.from_url(url) == VideoSource.AUTO
