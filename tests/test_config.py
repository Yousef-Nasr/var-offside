"""config.yaml loading and model-path lookup."""

from __future__ import annotations

from pathlib import Path

import pytest

from var_offside.config import apply_overrides, load_settings
from var_offside.config import paths
from var_offside.config.settings import Settings

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An empty project home, so lookups never see real weights."""
    monkeypatch.setenv("VAR_OFFSIDE_HOME", str(tmp_path))
    monkeypatch.delenv("VAR_OFFSIDE_WEIGHTS_DIR", raising=False)
    monkeypatch.delenv("VAR_OFFSIDE_RESULTS_DIR", raising=False)
    monkeypatch.delenv("VAR_OFFSIDE_CONFIG", raising=False)
    return tmp_path


def test_example_config_loads():
    settings = load_settings(REPO / "config.example.yaml")
    assert settings.server.port == 8765
    assert settings.server.host == "127.0.0.1"
    assert settings.calibration.green_lower == (35, 40, 40)
    assert settings.offside.field_length == 105.0


def test_overrides_apply_and_coerce(home):
    settings = apply_overrides(
        Settings(),
        {
            "device": "cpu",
            "server": {"port": 9000, "open_browser": False},
            "offside": {"field_length": 100},  # int in YAML, float in settings
            "visualization": {"team1_color": [1, 2, 3]},
        },
    )
    assert settings.device == "cpu"
    assert settings.server.port == 9000
    assert settings.server.open_browser is False
    assert settings.offside.field_length == 100.0
    assert isinstance(settings.offside.field_length, float)
    assert settings.visualization.team1_color == (1, 2, 3)


@pytest.mark.parametrize(
    "data, message",
    [
        ({"servr": {"port": 1}}, "unknown setting 'servr'"),
        ({"server": {"prot": 1}}, "unknown setting 'server.prot'"),
        ({"server": {"port": "eighty"}}, "server.port should be int"),
        ({"server": 8765}, "must be a mapping"),
    ],
)
def test_bad_config_is_rejected(home, data, message):
    with pytest.raises(ValueError, match=message):
        apply_overrides(Settings(), data)


def test_config_file_found_in_project_home(home):
    (home / "config.yaml").write_text("server:\n  port: 9100\n", encoding="utf-8")
    settings = load_settings()
    assert settings.server.port == 9100
    assert settings.config_file == home / "config.yaml"


def test_weights_dir_is_searched_first(home):
    (home / "weights").mkdir()
    (home / "weights" / "best_tuned.pt").write_bytes(b"x")
    (home / "best_tuned.pt").write_bytes(b"old layout")
    assert paths.resolve_weight("best_tuned.pt") == str(home / "weights" / "best_tuned.pt")


def test_old_layout_still_found(home):
    (home / "best_tuned.pt").write_bytes(b"x")
    assert paths.resolve_weight("best_tuned.pt") == str(home / "best_tuned.pt")


def test_missing_weight_name_passes_through(home):
    # Ultralytics downloads its official weights by name.
    assert paths.resolve_weight("yolo11x-pose.pt") == "yolo11x-pose.pt"
    assert paths.resolve_weight("resnet50") == "resnet50"


def test_first_existing_points_missing_files_at_weights(home):
    assert paths.first_existing("pnlcalib/SV_kp", "PnLCalib/weights/SV_kp") == str(
        home / "weights" / "pnlcalib" / "SV_kp"
    )
    legacy = home / "PnLCalib" / "weights"
    legacy.mkdir(parents=True)
    (legacy / "SV_kp").write_bytes(b"x")
    assert paths.first_existing("pnlcalib/SV_kp", "PnLCalib/weights/SV_kp") == str(
        legacy / "SV_kp"
    )


def test_third_party_dir_prefers_third_party(home):
    assert paths.third_party_dir("PnLCalib") == home / "third_party" / "PnLCalib"
    (home / "PnLCalib").mkdir()
    assert paths.third_party_dir("PnLCalib") == home / "PnLCalib"
    (home / "third_party" / "PnLCalib").mkdir(parents=True)
    assert paths.third_party_dir("PnLCalib") == home / "third_party" / "PnLCalib"


def test_weights_dir_env_override(home, tmp_path_factory, monkeypatch):
    elsewhere = tmp_path_factory.mktemp("models")
    (elsewhere / "best_tuned.pt").write_bytes(b"x")
    monkeypatch.setenv("VAR_OFFSIDE_WEIGHTS_DIR", str(elsewhere))
    assert paths.resolve_weight("best_tuned.pt") == str(elsewhere / "best_tuned.pt")
