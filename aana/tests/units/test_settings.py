# ruff: noqa: S101, S108
from pathlib import Path

from aana.configs.settings import Settings


def test_default_tmp_data_dir():
    """Test that the default temporary data directory is set correctly."""
    settings = Settings()
    assert settings.tmp_data_dir == Path("/tmp/aana_data")


def test_custom_tmp_data_dir(monkeypatch):
    """Test that the custom temporary data directory with environment variable is set correctly."""
    test_path = "/tmp/override/path"
    monkeypatch.setenv("TMP_DATA_DIR", test_path)
    settings = Settings()
    assert settings.tmp_data_dir == Path(test_path)


def test_changing_tmp_data_dir():
    """Test that changing the temporary data directory is reflected in the other directories."""
    new_tmp_data_dir = Path("/tmp/new_tmp_data_dir")
    settings = Settings(tmp_data_dir=new_tmp_data_dir)

    assert settings.tmp_data_dir == new_tmp_data_dir
    assert settings.image_dir == new_tmp_data_dir / "images"
    assert settings.video_dir == new_tmp_data_dir / "videos"
    assert settings.audio_dir == new_tmp_data_dir / "audios"
    assert settings.model_dir == new_tmp_data_dir / "models"

    # Check that we can change the image directory independently
    new_image_dir = Path("/tmp/new_image_dir")
    settings = Settings(tmp_data_dir=new_tmp_data_dir, image_dir=new_image_dir)
    assert settings.tmp_data_dir == new_tmp_data_dir
    assert settings.image_dir == new_image_dir

    # Check that we can change the video directory independently
    new_video_dir = Path("/tmp/new_video_dir")
    settings = Settings(tmp_data_dir=new_tmp_data_dir, video_dir=new_video_dir)
    assert settings.tmp_data_dir == new_tmp_data_dir
    assert settings.video_dir == new_video_dir

    # Check that we can change the audio directory independently
    new_audio_dir = Path("/tmp/new_audio_dir")
    settings = Settings(tmp_data_dir=new_tmp_data_dir, audio_dir=new_audio_dir)
    assert settings.tmp_data_dir == new_tmp_data_dir
    assert settings.audio_dir == new_audio_dir

    # Check that we can change the model directory independently
    new_model_dir = Path("/tmp/new_model_dir")
    settings = Settings(tmp_data_dir=new_tmp_data_dir, model_dir=new_model_dir)
    assert settings.tmp_data_dir == new_tmp_data_dir
    assert settings.model_dir == new_model_dir
