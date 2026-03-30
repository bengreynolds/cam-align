from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
from scipy.signal import butter, filtfilt, savgol_coeffs

from cam_align_tool.config.logging_utils import log_event
from cam_align_tool.core.models import InspectionResult, ScorerFolder, SessionMode
from cam_align_tool.core.transforms import atomic_replace

_LOG = logging.getLogger("cam_align_tool.core.regenerate")

_PELLET_BP = 2
_SIDE_HAND_BPS = (3, 4, 5)
_FRONT_HAND_BPS = (6, 7)

_PIX2MM_SIDECAM: float = 0.128865083
_PIX2MM_FRONTCAM: float = 0.2057297
_SG_WINDOW_LEN: int = 9
_SG_POLYORDER: int = 3
_BW_CUTOFF_FREQ: float = 50.0
_BW_FILTER_ORDER: int = 5


class TrajCol:
    Y = 0
    Y_FILT = 1
    Z = 2
    Z_FILT = 3
    YZ_LHOOD = 4
    X = 5
    X_FILT = 6
    X_LHOOD = 7
    DIST = 8
    SPEED = 9
    SPEED_FILT = 10


def _emit(progress: Optional[Callable[[str], None]], message: str) -> None:
    if callable(progress):
        progress(message)
    _LOG.info(message)


def _load_markers(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise RuntimeError(f"Unsupported scorer array shape for {path.name}: expected (N, 5), got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _dense_frame_count(markers: np.ndarray, label: str) -> int:
    frames = np.unique(markers[:, 0].astype(np.int64))
    if frames.size == 0:
        raise RuntimeError(f"{label}: scorer file contains no rows")
    expected = np.arange(int(frames[0]), int(frames[0]) + frames.size, dtype=np.int64)
    if not np.array_equal(frames, expected):
        raise RuntimeError(f"{label}: scorer file frame numbers are not dense and ordered from {int(frames[0])}")
    if int(frames[0]) != 0:
        raise RuntimeError(f"{label}: scorer file does not start at frame 0")
    return int(frames.size)


def _rows_for_part(markers: np.ndarray, part_value: int, n_frames: int, label: str) -> np.ndarray:
    rows = markers[markers[:, 1].astype(np.int64) == int(part_value)]
    if rows.shape[0] != n_frames:
        raise RuntimeError(
            f"{label}: expected {n_frames} rows for body part {part_value}, found {rows.shape[0]}"
        )
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    frames = rows[:, 0].astype(np.int64)
    if not np.array_equal(frames, np.arange(n_frames, dtype=np.int64)):
        raise RuntimeError(f"{label}: body part {part_value} rows do not cover every frame exactly once")
    return rows


def _best_hand_rows(markers: np.ndarray, part_values: Iterable[int], n_frames: int, label: str) -> np.ndarray:
    per_part = [_rows_for_part(markers, int(part_value), n_frames, label) for part_value in part_values]
    likelihoods = np.column_stack([rows[:, 4] for rows in per_part])
    best = np.argmax(likelihoods, axis=1)
    frame_idx = np.arange(n_frames)
    xs = np.column_stack([rows[:, 2] for rows in per_part])
    ys = np.column_stack([rows[:, 3] for rows in per_part])
    out = np.empty((n_frames, 3), dtype=np.float32)
    out[:, 0] = xs[frame_idx, best]
    out[:, 1] = ys[frame_idx, best]
    out[:, 2] = likelihoods[frame_idx, best]
    return out


def _safe_lowpass(vec: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    if vec.size < max(16, (3 * max(len(a), len(b))) + 1):
        return vec.astype(np.float32, copy=True)
    return filtfilt(b, a, vec).astype(np.float32, copy=False)


def _safe_speed_smooth(speed: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    min_len = max(_SG_WINDOW_LEN, (3 * len(coeffs)) + 1)
    if speed.size < min_len:
        return speed.astype(np.float32, copy=True)
    return filtfilt(coeffs, [1.0], speed).astype(np.float32, copy=False)


def _interpolate_low_confidence(values: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    lhood = likelihood.astype(np.float32, copy=True)
    if out.size == 0:
        return out
    lhood[0] = 1.0
    lhood[-1] = 1.0
    low_conf = lhood < 0.9
    if int(np.sum(low_conf)) >= out.size - 2:
        out[:] = 0.0
        return out
    frame_vec = np.arange(out.size, dtype=np.int64)
    out[:] = np.interp(frame_vec, frame_vec[~low_conf], out[~low_conf])
    out[np.isnan(out)] = 0.0
    return out


def _build_traj(x: np.ndarray, x_lhood: np.ndarray, y: np.ndarray, z: np.ndarray, yz_lhood: np.ndarray,
                frame_rate: int) -> np.ndarray:
    x = _interpolate_low_confidence(x, x_lhood)
    y = _interpolate_low_confidence(y, yz_lhood)
    z = _interpolate_low_confidence(z, yz_lhood)

    normalized_cutoff_freq = _BW_CUTOFF_FREQ / (0.5 * float(frame_rate))
    normalized_cutoff_freq = min(0.99, max(0.001, normalized_cutoff_freq))
    b, a = butter(_BW_FILTER_ORDER, normalized_cutoff_freq, btype="low", analog=False, output="ba")
    sg_coeffs = savgol_coeffs(_SG_WINDOW_LEN, _SG_POLYORDER)

    x_filt = _safe_lowpass(x, b, a)
    y_filt = _safe_lowpass(y, b, a)
    z_filt = _safe_lowpass(z, b, a)

    pos_change = np.sqrt(np.diff(x_filt) ** 2 + np.diff(y_filt) ** 2 + np.diff(z_filt) ** 2)
    pos_change = np.concatenate(([0.0], pos_change)).astype(np.float32, copy=False)
    speed = (pos_change * (float(frame_rate) / 1000.0)).astype(np.float32, copy=False)
    speed_filt = _safe_speed_smooth(speed, sg_coeffs)

    traj = np.empty((x.shape[0], 11), dtype=np.float32)
    traj[:, TrajCol.Y] = y
    traj[:, TrajCol.Y_FILT] = y_filt
    traj[:, TrajCol.Z] = z
    traj[:, TrajCol.Z_FILT] = z_filt
    traj[:, TrajCol.YZ_LHOOD] = yz_lhood
    traj[:, TrajCol.X] = x
    traj[:, TrajCol.X_FILT] = x_filt
    traj[:, TrajCol.X_LHOOD] = x_lhood
    traj[:, TrajCol.DIST] = pos_change
    traj[:, TrajCol.SPEED] = speed
    traj[:, TrajCol.SPEED_FILT] = speed_filt
    return traj


def regenerate_hand_pellet_for_scorer(
    scorer: ScorerFolder,
    inspection: InspectionResult,
    progress: Optional[Callable[[str], None]] = None,
) -> list[Path]:
    if inspection.mode != SessionMode.LEGACY:
        raise RuntimeError("Standalone hand/pellet regeneration is only implemented for legacy side/front sessions.")
    if not scorer.supports_hand_pellet_regen:
        reason = scorer.regen_note or "required scorer files are not available"
        raise RuntimeError(f"Scorer '{scorer.name}' does not support hand/pellet regeneration: {reason}")

    side_path = Path(scorer.path, "sideCam.npy")
    front_path = Path(scorer.path, "frontCam.npy")
    _emit(progress, f"Loading scorer camera files from {scorer.path.name}")
    side = _load_markers(side_path)
    front = _load_markers(front_path)

    n_frames_side = _dense_frame_count(side, side_path.name)
    n_frames_front = _dense_frame_count(front, front_path.name)
    if n_frames_side != n_frames_front:
        raise RuntimeError(
            f"Scorer '{scorer.name}' has mismatched camera frame counts: sideCam={n_frames_side}, frontCam={n_frames_front}"
        )
    n_frames = n_frames_side
    frame_rate = int(round(inspection.camera_files[inspection.master_camera].fps)) or 30

    pellet_side = _rows_for_part(side, _PELLET_BP, n_frames, side_path.name)
    pellet_front = _rows_for_part(front, _PELLET_BP, n_frames, front_path.name)
    hand_side = _best_hand_rows(side, _SIDE_HAND_BPS, n_frames, side_path.name)
    hand_front = _best_hand_rows(front, _FRONT_HAND_BPS, n_frames, front_path.name)

    _emit(progress, f"Filtering hand and pellet trajectories for scorer {scorer.name}")
    hand_y = hand_side[:, 0] * _PIX2MM_SIDECAM
    hand_z = hand_side[:, 1] * _PIX2MM_SIDECAM
    hand_yz_lhood = hand_side[:, 2]
    hand_x = hand_front[:, 0] * _PIX2MM_FRONTCAM
    hand_x_lhood = hand_front[:, 2]

    pellet_y = pellet_side[:, 2] * _PIX2MM_SIDECAM
    pellet_z = pellet_side[:, 3] * _PIX2MM_SIDECAM
    pellet_yz_lhood = pellet_side[:, 4]
    pellet_x = pellet_front[:, 2] * _PIX2MM_FRONTCAM
    pellet_x_lhood = pellet_front[:, 4]

    hand_traj = _build_traj(hand_x, hand_x_lhood, hand_y, hand_z, hand_yz_lhood, frame_rate)
    pellet_traj = _build_traj(pellet_x, pellet_x_lhood, pellet_y, pellet_z, pellet_yz_lhood, frame_rate)

    out_hand = Path(scorer.path, "hand.npy")
    out_pellet = Path(scorer.path, "pellet.npy")
    tmp_hand = Path(scorer.path, "hand.npy.tmp")
    tmp_pellet = Path(scorer.path, "pellet.npy.tmp")
    with tmp_hand.open("wb") as stream:
        np.save(stream, hand_traj)
    with tmp_pellet.open("wb") as stream:
        np.save(stream, pellet_traj)
    atomic_replace(tmp_hand, out_hand)
    atomic_replace(tmp_pellet, out_pellet)
    _emit(progress, f"Generated {out_hand.relative_to(inspection.root)}")
    _emit(progress, f"Generated {out_pellet.relative_to(inspection.root)}")
    log_event(
        _LOG,
        "regenerate_hand_pellet_for_scorer",
        root=inspection.root,
        scorer=scorer.name,
        frames=n_frames,
        frame_rate=frame_rate,
    )
    return [out_hand, out_pellet]
