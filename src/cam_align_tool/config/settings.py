from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


_SETTINGS_DIR = Path(".cam_align_state")
_SETTINGS_FILE = _SETTINGS_DIR / "settings.json"


@dataclass
class AppSettings:
    last_root: str = ""
    last_secondary_camera: str = ""
    last_offset: int = 0


def load_settings() -> AppSettings:
    try:
        raw = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
        return AppSettings(
            last_root=str(raw.get("last_root", "") or ""),
            last_secondary_camera=str(raw.get("last_secondary_camera", "") or ""),
            last_offset=int(raw.get("last_offset", 0) or 0),
        )
    except Exception:
        return AppSettings()


def save_settings(settings: AppSettings) -> None:
    _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    _SETTINGS_FILE.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
