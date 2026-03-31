from __future__ import annotations

import logging
import re
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

try:
    import cv2
except Exception:
    cv2 = None

from cam_align_tool.config.logging_utils import log_event
from cam_align_tool.core.models import CameraFile, InspectionResult, ReachEpoch, ScorerFolder, SessionMode

_LOG = logging.getLogger("cam_align_tool.core.inspect")

_LEGACY_CAMS = ["sideCam", "frontCam", "fastCam"]
_FIXED_CAMS = ["left", "right"]
_VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"]
_IGNORED_CAMERA_NAMES = {"stimcam", "cam3"}
_SUPPORTED_FRAME_RATES = {30, 60, 90, 120, 150, 180, 200, 240}
_SCORER_DIR_IGNORES = {".cam_align_backup", ".cam_align_state", ".git", "__pycache__"}


def _is_ignored_camera_name(name: str) -> bool:
    return str(name).strip().lower() in _IGNORED_CAMERA_NAMES


def _best_video_for_camera(root: Path, camera: str) -> Optional[Path]:
    camera_lower = camera.lower()
    candidates = [
        p for p in root.glob("*")
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTS and camera_lower in p.name.lower()
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


def _session_prefix(root: Path, camera_names: Optional[list[str]] = None) -> str:
    names = [p.stem for p in root.iterdir() if p.is_file() and p.suffix.lower() in _VIDEO_EXTS]
    tokens = [token for token in (camera_names or []) if token and not _is_ignored_camera_name(token)]
    tokens.extend(_LEGACY_CAMS + _FIXED_CAMS)
    unique_tokens: list[str] = []
    seen_tokens: set[str] = set()
    for token in tokens:
        key = token.lower()
        if key in seen_tokens:
            continue
        seen_tokens.add(key)
        unique_tokens.append(token)
    for stem in sorted(names):
        for token in unique_tokens:
            key = token.lower()
            idx = stem.lower().find(key)
            if idx > 0:
                return stem[:idx]
    return root.name + "_"


def _find_timestamp(root: Path, prefix: str, camera: str) -> Optional[Path]:
    direct = root / f"{prefix}{camera}_timestamps.txt"
    if direct.is_file():
        return direct
    matches = sorted([p for p in root.glob(f"*{camera}*_timestamps.txt") if p.is_file()])
    return matches[0] if matches else None


def _find_events_file(root: Path, prefix: str) -> Optional[Path]:
    direct = root / f"{prefix}events.txt"
    if direct.is_file():
        return direct
    matches = sorted([p for p in root.glob("*events.txt") if p.is_file()])
    return matches[0] if matches else None


def _find_reaches_file(root: Path, prefix: str) -> Optional[Path]:
    direct = root / f"{prefix}reaches.txt"
    if direct.is_file():
        return direct
    matches = sorted(
        [
            p for p in root.glob("*reaches.txt")
            if p.is_file() and p.name.lower() != "detected_reaches.txt"
        ]
    )
    return matches[0] if matches else None


def _scan_scorer_folders(root: Path, mode: SessionMode) -> tuple[ScorerFolder, ...]:
    required = ("left.npy", "right.npy") if mode == SessionMode.FIXED_CAM else ("sideCam.npy", "frontCam.npy")
    out: list[ScorerFolder] = []
    for path in sorted([p for p in root.iterdir() if p.is_dir()]):
        if path.name.lower() in _SCORER_DIR_IGNORES:
            continue
        present = tuple(name for name in required if Path(path, name).is_file())
        if len(present) == 0:
            continue
        supports = mode == SessionMode.LEGACY and len(present) == len(required)
        note = ""
        if len(present) != len(required):
            note = f"missing required camera scorer files: {sorted(set(required) - set(present))}"
        elif mode == SessionMode.FIXED_CAM:
            note = "hand/pellet regeneration is not implemented for fixed-cam scorer outputs"
        out.append(
            ScorerFolder(
                name=path.name,
                path=path,
                camera_files=present,
                has_detected_markers=Path(path, "detected_markers.npy").is_file(),
                has_hand=Path(path, "hand.npy").is_file(),
                has_pellet=Path(path, "pellet.npy").is_file(),
                supports_hand_pellet_regen=supports,
                regen_note=note,
            )
        )
    return tuple(out)


def _camera_mode(videos: dict[str, Path]) -> SessionMode:
    if any(cam.lower() in {"left", "leftcam", "right", "rightcam"} for cam in videos):
        return SessionMode.FIXED_CAM
    return SessionMode.LEGACY


def _systemdata_path(root: Path, prefix: str) -> Optional[Path]:
    preferred = root / f"{prefix}systemdata_copy.yaml"
    if preferred.is_file():
        return preferred
    matches = sorted(root.glob("*systemdata_copy.yaml"))
    return matches[0] if matches else None


def _camera_configs_from_systemdata(root: Path, prefix: str) -> list[tuple[str, str, bool]]:
    path = _systemdata_path(root, prefix)
    if path is None:
        return []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, dict):
        return []

    out: list[tuple[str, str, bool]] = []
    for cam_key, cam_cfg in raw.items():
        if not isinstance(cam_cfg, dict):
            continue
        cam_key_str = str(cam_key).strip()
        if not cam_key_str.lower().startswith("cam"):
            continue
        if cam_key_str.lower() == "cam3":
            continue
        nickname = str(cam_cfg.get("nickname", "") or "").strip()
        if not nickname or _is_ignored_camera_name(nickname):
            continue
        out.append((cam_key_str, nickname, bool(cam_cfg.get("ismaster", False))))
    return out


def _configured_camera_names(root: Path, prefix: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for _cam_key, nickname, _is_master in _camera_configs_from_systemdata(root, prefix):
        key = nickname.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(nickname)
    return out


def _master_from_systemdata(root: Path, prefix: str) -> Optional[str]:
    for _cam_key, nickname, is_master in _camera_configs_from_systemdata(root, prefix):
        if is_master:
            return nickname
    return None


def _fallback_camera_names(root: Path) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for cam in _LEGACY_CAMS + _FIXED_CAMS:
        if _is_ignored_camera_name(cam):
            continue
        if _best_video_for_camera(root, cam) is None:
            continue
        key = cam.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cam)
    return out


def _resolve_master(videos: dict[str, Path], root: Path, prefix: str, mode: SessionMode) -> str:
    configured_master = _master_from_systemdata(root, prefix)
    if configured_master in videos:
        return configured_master
    default_master = "left" if mode == SessionMode.FIXED_CAM else "sideCam"
    if default_master in videos:
        return default_master
    if len(videos) == 2:
        return sorted(videos.keys())[0]
    raise RuntimeError("Unable to resolve a non-cam3 master camera from systemdata_copy.yaml and detected videos.")


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
            ffprobe = _resolve_ffprobe_path()
            if ffprobe is not None:
                fps = _ffprobe_fps(path, ffprobe)
        if fps <= 0:
            raise RuntimeError(f"Unable to determine frame rate for video: {path}")
        return frame_count, fps, width, height
    finally:
        cap.release()


def _resolve_ffprobe_path() -> Optional[str]:
    path = shutil.which("ffprobe")
    if path:
        return path
    candidates = [
        Path.home() / "AppData/Local/Microsoft/WinGet/Links/ffprobe.exe",
        Path("C:/Program Files/CBD/CHITUBOX_Basic/Resources/DependentSoftware/recordOrShot/ffprobe.exe"),
    ]
    winget_root = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    candidates.extend(sorted(winget_root.glob("Gyan.FFmpeg_*/*/bin/ffprobe.exe")))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _ffprobe_fps(path: Path, ffprobe_exe: str) -> float:
    import json
    import subprocess

    cmd = [
        ffprobe_exe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(out.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        return 0.0
    rate = str(streams[0].get("avg_frame_rate", "") or "").strip()
    if not rate or rate in {"0/0", "N/A"}:
        return 0.0
    try:
        return float(Fraction(rate))
    except Exception:
        return 0.0


def _load_timestamp_intervals(path: Path) -> np.ndarray:
    loaders = (
        lambda: np.loadtxt(path, dtype="i8", usecols=(0,), delimiter=","),
        lambda: np.loadtxt(path, dtype="i8", usecols=(0,)),
    )
    last_exc: Optional[Exception] = None
    for loader in loaders:
        try:
            arr = np.atleast_1d(loader())
            if arr.ndim != 1:
                raise RuntimeError("Timestamp file did not parse as a 1D interval array")
            return arr.astype(np.int64, copy=False)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Unable to parse timestamp intervals from {path}: {last_exc}")


def _detect_dropped_frames(timestamp_path: Path, captured_frame_count: int) -> tuple[tuple[int, ...], tuple[str, ...]]:
    warnings: list[str] = []
    intervals = _load_timestamp_intervals(timestamp_path)
    if intervals.size != captured_frame_count:
        warnings.append(
            f"{timestamp_path.name}: timestamp row count {int(intervals.size)} does not match captured video frames {captured_frame_count}"
        )
    if intervals.size == 0:
        return (), tuple(warnings)

    mean_interval = float(np.mean(intervals))
    if mean_interval <= 0:
        warnings.append(f"{timestamp_path.name}: invalid non-positive mean timestamp interval")
        return (), tuple(warnings)

    inferred_rate = int(np.round(1.0e9 / mean_interval))
    if inferred_rate not in _SUPPORTED_FRAME_RATES:
        warnings.append(f"{timestamp_path.name}: inferred frame rate {inferred_rate} Hz is unusual")

    nominal_interval = int(0.5 + 1.0e9 / inferred_rate)
    interval_diffs = np.abs(intervals - nominal_interval)
    drop_indices = np.nonzero(interval_diffs > int(nominal_interval / 2))[0]
    n_consec = np.round(interval_diffs[drop_indices] / nominal_interval).astype(int)

    dropped_frames: list[int] = []
    n_drops = 0
    for idx, drops_here in zip(drop_indices.tolist(), n_consec.tolist()):
        while drops_here > 0:
            dropped_frames.append(int(idx + 1 + n_drops))
            n_drops += 1
            drops_here -= 1
    return tuple(dropped_frames), tuple(warnings)


def _parse_pellet_delivery_frames(events_path: Path) -> tuple[int, ...]:
    pellet_frames: list[int] = []
    for raw_line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = re.split(r"[\t, ]+", line)
        if len(parts) < 2:
            continue
        if parts[0].strip().lower() != "pellet_delivery":
            continue
        try:
            pellet_frames.append(int(float(parts[1])))
        except Exception:
            continue
    return tuple(pellet_frames)


def _parse_reach_frames(reaches_path: Path) -> tuple[int, ...]:
    reach_frames: list[int] = []
    for raw_line in reaches_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 1:
            continue
        try:
            reach_frames.append(int(float(parts[0])))
        except Exception:
            continue
    return tuple(reach_frames)


def _build_reach_epochs(root: Path, prefix: str) -> tuple[tuple[ReachEpoch, ...], tuple[str, ...]]:
    events_path = _find_events_file(root, prefix)
    reaches_path = _find_reaches_file(root, prefix)
    warnings: list[str] = []
    if events_path is None:
        if reaches_path is not None:
            warnings.append(f"{reaches_path.name}: reaches file found but matching events.txt was not found, so pellet epoch navigation is unavailable")
        return (), tuple(warnings)

    pellet_frames = _parse_pellet_delivery_frames(events_path)
    if len(pellet_frames) == 0:
        warnings.append(f"{events_path.name}: no pellet_delivery entries were found, so pellet epoch navigation is unavailable")
        return (), tuple(warnings)

    reach_frames = _parse_reach_frames(reaches_path) if reaches_path is not None else ()
    epochs: list[ReachEpoch] = []
    for idx, pellet_frame in enumerate(pellet_frames, start=1):
        next_pellet = pellet_frames[idx] if idx < len(pellet_frames) else None
        if next_pellet is None:
            epoch_reaches = tuple(frame for frame in reach_frames if frame >= pellet_frame)
        else:
            epoch_reaches = tuple(frame for frame in reach_frames if pellet_frame <= frame < next_pellet)
        label = f"Epoch {idx}: pellet_delivery @ {pellet_frame}"
        if reaches_path is not None:
            if epoch_reaches:
                label += f" ({len(epoch_reaches)} reach{'es' if len(epoch_reaches) != 1 else ''})"
            else:
                label += " (no reaches)"
        epochs.append(
            ReachEpoch(
                index=idx,
                pellet_frame=int(pellet_frame),
                reach_frames=epoch_reaches,
                label=label,
            )
        )
    return tuple(epochs), tuple(warnings)


def inspect_input_folder(root: Path) -> InspectionResult:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Input folder not found: {root}")

    preliminary_prefix = _session_prefix(root)
    configured_names = _configured_camera_names(root, preliminary_prefix)
    prefix = _session_prefix(root, configured_names)

    candidate_names = configured_names or _fallback_camera_names(root)
    videos: dict[str, Path] = {}
    for cam in candidate_names:
        if _is_ignored_camera_name(cam):
            continue
        video = _best_video_for_camera(root, cam)
        if video is not None:
            videos[cam] = video
    if len(videos) < 2:
        raise RuntimeError("Could not detect at least two non-cam3 camera videos in the selected folder.")

    mode = _camera_mode(videos)
    master = _resolve_master(videos, root, prefix, mode)
    if master not in videos:
        raise RuntimeError(f"Master camera from systemdata_copy.yaml was not found among detected videos: {master}")

    camera_files: dict[str, CameraFile] = {}
    warnings: list[str] = []
    for cam, video_path in videos.items():
        frame_count, fps, width, height = _video_info(video_path)
        timestamp_path = _find_timestamp(root, prefix, cam)
        dropped_frames: tuple[int, ...] = ()
        camera_warnings: list[str] = []
        if timestamp_path is not None:
            try:
                dropped_frames, timestamp_warnings = _detect_dropped_frames(timestamp_path, frame_count)
                camera_warnings.extend(timestamp_warnings)
            except Exception as exc:
                camera_warnings.append(f"{cam}: timestamp inspection failed: {exc}")
        if len(dropped_frames) > 0:
            camera_warnings.append(
                f"{cam}: detected {len(dropped_frames)} dropped frame(s) from timestamps during recording; this tool only fixes startup/end offset mismatch"
            )
        warnings.extend(camera_warnings)
        camera_files[cam] = CameraFile(
            camera=cam,
            video_path=video_path,
            frame_count=frame_count,
            fps=fps,
            width=width,
            height=height,
            timestamp_path=timestamp_path,
            dropped_frames=dropped_frames,
            inspection_warnings=tuple(camera_warnings),
        )

    all_files = sorted(
        [
            p for p in root.rglob("*")
            if p.is_file() and ".cam_align_backup" not in {part.lower() for part in p.parts}
        ]
    )
    scorer_folders = _scan_scorer_folders(root, mode)
    reach_epochs, epoch_warnings = _build_reach_epochs(root, prefix)
    warnings.extend(epoch_warnings)
    cameras = sorted(camera_files.keys())
    log_event(
        _LOG,
        "inspect_input_folder",
        root=root,
        mode=mode.value,
        master=master,
        cameras="|".join(cameras),
        warnings=len(warnings),
        reach_epochs=len(reach_epochs),
        scorer_folders=len(scorer_folders),
    )
    return InspectionResult(
        root=root,
        mode=mode,
        master_camera=master,
        cameras=cameras,
        camera_files=camera_files,
        session_prefix=prefix,
        all_files=all_files,
        warnings=tuple(warnings),
        reach_epochs=reach_epochs,
        reaches_file_present=_find_reaches_file(root, prefix) is not None,
        scorer_folders=scorer_folders,
    )
