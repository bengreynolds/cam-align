from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


_SETTINGS_DIR = Path(".cam_align_state")
_SETTINGS_FILE = _SETTINGS_DIR / "settings.json"


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
    try:
        raw = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
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
        return AppSettings()


def save_settings(settings: AppSettings) -> None:
    _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    _SETTINGS_FILE.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
