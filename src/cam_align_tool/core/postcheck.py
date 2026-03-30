from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd

from cam_align_tool.core.models import InspectionResult, PostCheckReport

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
_PAIRABLE_EXTS = {".csv", ".h5", ".npy", ".txt"}


def _ffprobe_path() -> str:
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
    raise RuntimeError("ffprobe was not found on PATH. Install FFmpeg so post-processing video checks can run.")


def _ffprobe_video_frame_count(path: Path) -> int:
    cmd = [
        _ffprobe_path(),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(out.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe returned no video stream for {path.name}")
    stream = streams[0]
    for key in ("nb_read_frames", "nb_frames"):
        value = str(stream.get(key, "") or "").strip()
        if value.isdigit():
            return int(value)
    raise RuntimeError(f"ffprobe did not report a usable frame count for {path.name}")


def _cv2_video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open {path.name}")
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def _cv2_sample_read_ok(path: Path, frame_indices: list[int]) -> bool:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open {path.name}")
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                return False
        return True
    finally:
        cap.release()


def _csv_row_count(path: Path) -> int:
    best_rows: Optional[int] = None
    for header in ([0, 1, 2], [0, 1], [0]):
        try:
            rows = int(pd.read_csv(path, header=header, index_col=0).shape[0])
            if best_rows is None or rows > best_rows:
                best_rows = rows
        except Exception:
            continue
    if best_rows is None:
        raise RuntimeError(f"Unable to parse CSV row count for {path.name}")
    return best_rows


def _h5_row_count(path: Path) -> int:
    with pd.HDFStore(str(path), mode="r") as store:
        keys = list(store.keys())
        if not keys:
            raise RuntimeError(f"No HDF keys found in {path.name}")
        return int(store[keys[0]].shape[0])


def _npy_frame_count(path: Path) -> int:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 2 and arr.shape[1] == 5:
        return int(len(np.unique(arr[:, 0])))
    if arr.ndim == 0:
        raise RuntimeError(f"Scalar NPY is not a pairable frame series: {path.name}")
    return int(arr.shape[0])


def _timestamps_row_count(path: Path) -> int:
    return sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))


def _dense_length_for_path(path: Path) -> int:
    name_lower = path.name.lower()
    if path.suffix.lower() in _VIDEO_EXTS:
        return _ffprobe_video_frame_count(path)
    if name_lower.endswith("_timestamps.txt"):
        return _timestamps_row_count(path)
    if path.suffix.lower() == ".csv":
        return _csv_row_count(path)
    if path.suffix.lower() == ".h5":
        return _h5_row_count(path)
    if path.suffix.lower() == ".npy":
        return _npy_frame_count(path)
    raise RuntimeError(f"Unsupported pairable file type: {path.name}")


def _pair_key(relpath: Path, master: str, secondary: str) -> Optional[str]:
    rel_lower = str(relpath).lower()
    has_master = master.lower() in rel_lower
    has_secondary = secondary.lower() in rel_lower
    if has_master == has_secondary:
        return None
    return rel_lower.replace(master.lower(), "{cam}").replace(secondary.lower(), "{cam}")


def _pairable_path(path: Path) -> bool:
    return path.suffix.lower() in _PAIRABLE_EXTS


def _paired_dense_artifacts(inspection: InspectionResult, secondary_camera: str) -> list[tuple[Path, Path]]:
    master = inspection.master_camera
    buckets: dict[str, dict[str, Path]] = {}
    for path in inspection.all_files:
        if not _pairable_path(path):
            continue
        relpath = path.relative_to(inspection.root)
        rel_lower = str(relpath).lower()
        has_master = master.lower() in rel_lower
        has_secondary = secondary_camera.lower() in rel_lower
        if has_master == has_secondary:
            continue
        key = _pair_key(relpath, master, secondary_camera)
        if key is None:
            continue
        name_lower = path.name.lower()
        if path.suffix.lower() == ".txt" and not name_lower.endswith("_timestamps.txt"):
            continue
        which = "master" if has_master else "secondary"
        bucket = buckets.setdefault(key, {})
        bucket[which] = path
    out: list[tuple[Path, Path]] = []
    for bucket in buckets.values():
        if "master" in bucket and "secondary" in bucket:
            out.append((bucket["master"], bucket["secondary"]))
    return sorted(out, key=lambda pair: str(pair[0]).lower())


def run_post_process_check(inspection: InspectionResult, secondary_camera: str,
                           progress: Optional[Callable[[str], None]] = None) -> PostCheckReport:
    if secondary_camera == inspection.master_camera:
        raise RuntimeError("Post-processing check requires a secondary camera.")
    if secondary_camera not in inspection.camera_files:
        raise RuntimeError(f"Secondary camera not found: {secondary_camera}")

    def emit(msg: str) -> None:
        if callable(progress):
            progress(msg)

    master = inspection.master_camera
    lines = [
        "Post-Processing Check",
        f"Root: {inspection.root}",
        f"Master: {master}",
        f"Secondary: {secondary_camera}",
        "",
    ]
    failures: list[str] = []

    master_video = inspection.camera_files[master].video_path
    secondary_video = inspection.camera_files[secondary_camera].video_path
    emit("Checking video lengths with ffprobe and OpenCV.")
    master_ffprobe = _ffprobe_video_frame_count(master_video)
    secondary_ffprobe = _ffprobe_video_frame_count(secondary_video)
    lines.append("Video checks:")
    lines.append(f"  - ffprobe frame counts: {master_video.name}={master_ffprobe}, {secondary_video.name}={secondary_ffprobe}")
    if master_ffprobe != secondary_ffprobe:
        failures.append(f"Video frame count mismatch: {master_ffprobe} vs {secondary_ffprobe}")

    master_cv2 = _cv2_video_frame_count(master_video)
    secondary_cv2 = _cv2_video_frame_count(secondary_video)
    lines.append(f"  - OpenCV frame counts: {master_video.name}={master_cv2}, {secondary_video.name}={secondary_cv2}")
    if master_cv2 != master_ffprobe:
        failures.append(f"OpenCV/ffprobe mismatch for {master_video.name}: {master_cv2} vs {master_ffprobe}")
    if secondary_cv2 != secondary_ffprobe:
        failures.append(f"OpenCV/ffprobe mismatch for {secondary_video.name}: {secondary_cv2} vs {secondary_ffprobe}")

    sample_count = min(master_ffprobe, secondary_ffprobe)
    sample_indices = sorted({0, max(0, sample_count // 2), max(0, sample_count - 1)}) if sample_count > 0 else []
    if sample_indices:
        master_reads = _cv2_sample_read_ok(master_video, sample_indices)
        secondary_reads = _cv2_sample_read_ok(secondary_video, sample_indices)
        lines.append(f"  - Sample frame reads at {sample_indices}: {master}={master_reads}, {secondary_camera}={secondary_reads}")
        if not master_reads:
            failures.append(f"OpenCV sample frame reads failed for {master_video.name}")
        if not secondary_reads:
            failures.append(f"OpenCV sample frame reads failed for {secondary_video.name}")
    lines.append("")

    emit("Checking dense paired side/front or left/right artifacts.")
    lines.append("Dense pair checks:")
    pair_found = False
    for master_path, secondary_path in _paired_dense_artifacts(inspection, secondary_camera):
        pair_found = True
        master_len = _dense_length_for_path(master_path)
        secondary_len = _dense_length_for_path(secondary_path)
        rel_master = master_path.relative_to(inspection.root)
        rel_secondary = secondary_path.relative_to(inspection.root)
        ok = master_len == secondary_len
        lines.append(
            f"  - {rel_master} | {rel_secondary}: {master_len} vs {secondary_len} {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append(f"Paired length mismatch: {rel_master} vs {rel_secondary} => {master_len} vs {secondary_len}")
    if not pair_found:
        lines.append("  - No dense side/front or left/right file pairs were found.")

    lines.append("")
    if inspection.warnings:
        lines.append("Inspection warnings:")
        lines.extend([f"  - {warning}" for warning in inspection.warnings])
        lines.append("")

    passed = len(failures) == 0
    lines.append(f"Overall result: {'PASS' if passed else 'FAIL'}")
    if failures:
        lines.append("Failures:")
        lines.extend([f"  - {failure}" for failure in failures])
    return PostCheckReport(passed=passed, details="\n".join(lines))
