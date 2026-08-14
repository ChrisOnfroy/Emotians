from pathlib import Path

import pytest

from emotians.config import load_config


def test_loads_stage_one_config() -> None:
    config = load_config(Path("configs/preview.yaml"))

    assert config.camera.index == 0
    assert config.preview.roi_width_ratio == 0.42
    assert config.preview.quit_keys == ("q", "ESC")


def test_rejects_unknown_config_key(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("camera:\n  invalid_key: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown camera configuration"):
        load_config(config_path)
