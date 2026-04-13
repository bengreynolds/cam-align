from __future__ import annotations

import os
import json
from dataclasses import asdict, dataclass
from pathlib import Path


_LEGACY_SETTINGS_DIR = Path(".cam_align_state")
_SETTINGS_FILE_NAME = "settings.json"


def _state_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "cam-align"

    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home) / "cam-align"

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home) / "cam-align"

    return Path.home() / ".cam_align"


def _settings_file() -> Path:
    return _state_dir() / _SETTINGS_FILE_NAME


def _legacy_settings_file() -> Path:
    return _LEGACY_SETTINGS_DIR / _SETTINGS_FILE_NAME


@dataclass
class AppSettings:
    last_root: str = ""
    default_browse_root: str = ""
    last_secondary_camera: str = ""
    last_offset: int = 0
    ffprobe_path: str = ""
    create_backup: bool = True
    auto_post_check: bool = False


def load_settings() -> AppSettings:
    for path in (_settings_file(), _legacy_settings_file()):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return AppSettings(
                last_root=str(raw.get("last_root", "") or ""),
                default_browse_root=str(raw.get("default_browse_root", "") or ""),
                last_secondary_camera=str(raw.get("last_secondary_camera", "") or ""),
                last_offset=int(raw.get("last_offset", 0) or 0),
                ffprobe_path=str(raw.get("ffprobe_path", "") or ""),
                create_backup=bool(raw.get("create_backup", True)),
                auto_post_check=bool(raw.get("auto_post_check", False)),
            )
        except Exception:
            continue
    return AppSettings()


def save_settings(settings: AppSettings) -> None:
    path = _settings_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
