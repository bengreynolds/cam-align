from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class SessionMode(str, Enum):
    LEGACY = "legacy"
    FIXED_CAM = "fixed-cam"


class TransformKind(str, Enum):
    VIDEO = "video"
    TIMESTAMPS = "timestamps"
    DATAFRAME_H5 = "dataframe_h5"
    DATAFRAME_CSV = "dataframe_csv"
    FRAME_LIST = "frame_list"
    SCORER_OUTPUT_NPY = "scorer_output_npy"
    DETECTED_MARKERS_NPY = "detected_markers_npy"
    NPY_TIMESERIES = "npy_timeseries"


class ChangeAction(str, Enum):
    REWRITE = "rewrite"
    INVALIDATE = "invalidate"
    SKIP = "skip"


@dataclass(frozen=True)
class CameraFile:
    camera: str
    video_path: Path
    frame_count: int
    fps: float
    width: int
    height: int
    timestamp_path: Optional[Path] = None
    dropped_frames: tuple[int, ...] = ()
    inspection_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Artifact:
    path: Path
    action: ChangeAction
    reason: str
    kind: Optional[TransformKind] = None
    camera: Optional[str] = None
    row_count: Optional[int] = None


@dataclass(frozen=True)
class InspectionResult:
    root: Path
    mode: SessionMode
    master_camera: str
    cameras: list[str]
    camera_files: dict[str, CameraFile]
    session_prefix: str
    all_files: list[Path]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlannedChange:
    artifact: Artifact
    backup_relpath: Path


@dataclass(frozen=True)
class CompensationPlan:
    inspection: InspectionResult
    secondary_camera: str
    offset: int
    target_frame_count: int
    rewrites: list[PlannedChange] = field(default_factory=list)
    invalidations: list[PlannedChange] = field(default_factory=list)
    skips: list[Artifact] = field(default_factory=list)


@dataclass(frozen=True)
class PlanSummary:
    rewrite_count: int
    invalidate_count: int
    skip_count: int
    details: str
