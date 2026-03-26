from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import yaml

try:
    import cv2
except Exception:
    cv2 = None

from cam_align_tool.config.logging_utils import log_event
from cam_align_tool.core.models import CameraFile, InspectionResult, SessionMode

_LOG = logging.getLogger("cam_align_tool.core.inspect")

_LEGACY_CAMS = ["sideCam", "frontCam", "stimCam", "fastCam"]
_FIXED_CAMS = ["left", "right"]
_VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"]


def _best_video_for_camera(root: Path, camera: str) -> Optional[Path]:
    candidates = [
        p for p in root.glob("*")
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTS and camera.lower() in p.name.lower()
    ]
    if len(candidates) == 0:
        return None
    if len(candidates) == 1:
        return candidates[0]
    best = None
    best_idx = -1
    for p in sorted(candidates):
        match = re.search(r"-(\d{4})\.[^.]+$", p.name)
        idx = int(match.group(1)) if match else 0
        if idx >= best_idx:
            best_idx = idx
            best = p
    return best or sorted(candidates)[-1]


def _session_prefix(root: Path) -> str:
    names = [p.stem for p in root.iterdir() if p.is_file() and p.suffix.lower() in _VIDEO_EXTS]
    for stem in sorted(names):
        for token in _LEGACY_CAMS + _FIXED_CAMS:
            idx = stem.lower().find(token.lower())
            if idx > 0:
                return stem[:idx]
    return root.name + "_"


def _find_timestamp(root: Path, prefix: str, camera: str) -> Optional[Path]:
    direct = root / f"{prefix}{camera}_timestamps.txt"
    if direct.is_file():
        return direct
    matches = sorted([p for p in root.glob(f"*{camera}*_timestamps.txt") if p.is_file()])
    return matches[0] if matches else None


def _camera_mode(videos: dict[str, Path]) -> SessionMode:
    if any(cam in videos for cam in _FIXED_CAMS):
        return SessionMode.FIXED_CAM
    return SessionMode.LEGACY


def _master_from_systemdata(root: Path, prefix: str) -> Optional[str]:
    candidates = [
        root / f"{prefix}systemdata_copy.yaml",
        root / f"{prefix}userdata_copy.yaml",
    ]
    for p in candidates:
        if not p.is_file():
            continue
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        for _, cam_cfg in raw.items():
            if isinstance(cam_cfg, dict) and bool(cam_cfg.get("ismaster", False)):
                nickname = str(cam_cfg.get("nickname", "") or "").strip()
                if nickname:
                    return nickname
    return None


def _video_info(path: Path) -> tuple[int, float, int, int]:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for video inspection.")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0:
            fps = 30.0
        return frame_count, fps, width, height
    finally:
        cap.release()


def inspect_input_folder(root: Path) -> InspectionResult:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Input folder not found: {root}")

    videos: dict[str, Path] = {}
    for cam in _LEGACY_CAMS + _FIXED_CAMS:
        video = _best_video_for_camera(root, cam)
        if video is not None:
            videos[cam] = video
    if len(videos) < 2:
        raise RuntimeError("Could not detect at least two camera videos in the selected folder.")

    prefix = _session_prefix(root)
    mode = _camera_mode(videos)
    default_master = "left" if mode == SessionMode.FIXED_CAM else "sideCam"
    master = _master_from_systemdata(root, prefix) or default_master
    if master not in videos:
        master = default_master if default_master in videos else sorted(videos.keys())[0]

    camera_files: dict[str, CameraFile] = {}
    for cam, video_path in videos.items():
        frame_count, fps, width, height = _video_info(video_path)
        camera_files[cam] = CameraFile(
            camera=cam,
            video_path=video_path,
            frame_count=frame_count,
            fps=fps,
            width=width,
            height=height,
            timestamp_path=_find_timestamp(root, prefix, cam),
        )

    all_files = sorted(
        [
            p for p in root.rglob("*")
            if p.is_file() and ".cam_align_backup" not in {part.lower() for part in p.parts}
        ]
    )
    cameras = sorted(camera_files.keys())
    log_event(
        _LOG,
        "inspect_input_folder",
        root=root,
        mode=mode.value,
        master=master,
        cameras="|".join(cameras),
    )
    return InspectionResult(
        root=root,
        mode=mode,
        master_camera=master,
        cameras=cameras,
        camera_files=camera_files,
        session_prefix=prefix,
        all_files=all_files,
    )
