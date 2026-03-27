from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import cv2
except Exception:
    cv2 = None

from cam_align_tool.core.models import TransformKind


def shift_video_file(src: Path, dst: Path, offset: int, output_frame_count: Optional[int] = None) -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for video rewriting.")
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for rewrite: {src}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0:
            fps = 30.0

        frames: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
        if len(frames) == 0:
            raise RuntimeError(f"No decodable frames found in video: {src}")

        target_count = len(frames) if output_frame_count is None else int(output_frame_count)
        if target_count < 0:
            raise RuntimeError(f"Negative output frame count requested for {src}: {target_count}")

        shifted: list[np.ndarray] = []
        for idx in range(target_count):
            src_idx = idx + int(offset)
            if src_idx < 0:
                src_idx = 0
            elif src_idx >= len(frames):
                src_idx = len(frames) - 1
            shifted.append(frames[src_idx])

        fourcc_tags = ["mp4v", "avc1"] if dst.suffix.lower() == ".mp4" else ["XVID", "MJPG"]
        writer = None
        for tag in fourcc_tags:
            writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*tag), fps, (width, height))
            if writer.isOpened():
                break
            writer.release()
            writer = None
        if writer is None:
            raise RuntimeError(f"Unable to create output video: {dst}")
        try:
            for frame in shifted:
                writer.write(frame)
        finally:
            writer.release()
    finally:
        cap.release()


def _sentinel_value(col: Any, dtype) -> Any:
    parts = [str(x).lower() for x in col] if isinstance(col, tuple) else [str(col).lower()]
    name = "|".join(parts)
    if "likelihood" in name or name.endswith("|p") or name == "p":
        return -1.0
    if pd.api.types.is_bool_dtype(dtype):
        return False
    if pd.api.types.is_integer_dtype(dtype):
        return -1
    if pd.api.types.is_numeric_dtype(dtype):
        return np.nan
    return ""


def shift_dataframe(df: pd.DataFrame, offset: int, output_rows: Optional[int] = None) -> pd.DataFrame:
    if df.shape[0] == 0:
        return df.copy(deep=True)
    target_rows = df.shape[0] if output_rows is None else int(output_rows)
    if target_rows < 0:
        raise RuntimeError(f"Negative output row count requested: {target_rows}")
    out = df.iloc[:target_rows].copy(deep=True)
    fill = {col: _sentinel_value(col, df[col].dtype) for col in df.columns}
    for row in range(target_rows):
        src_row = row + int(offset)
        if 0 <= src_row < df.shape[0]:
            out.iloc[row] = df.iloc[src_row]
        else:
            out.iloc[row] = pd.Series(fill)
    return out


def _open_hdf(path: Path) -> tuple[pd.DataFrame, str]:
    with pd.HDFStore(str(path), mode="r") as store:
        keys = list(store.keys())
        if len(keys) == 0:
            raise RuntimeError(f"No HDF keys found in {path}")
        key = keys[0]
    return pd.read_hdf(path, key=key), key


def shift_dataframe_h5(path: Path, dst: Path, offset: int, output_rows: Optional[int] = None) -> None:
    df, key = _open_hdf(path)
    shifted = shift_dataframe(df, offset, output_rows=output_rows)
    shifted.to_hdf(str(dst), key=key, format="table", mode="w")


def _try_read_csv(path: Path) -> tuple[pd.DataFrame, Optional[list[int]]]:
    attempts = ([0, 1, 2], [0, 1], [0])
    best_df = None
    best_header = None
    for header in attempts:
        try:
            df = pd.read_csv(path, header=header, index_col=0)
            if best_df is None or df.shape[0] > best_df.shape[0]:
                best_df = df
                best_header = list(header)
        except Exception:
            continue
    if best_df is not None and best_df.shape[0] > 0:
        return best_df, best_header
    raise RuntimeError(f"Unable to parse CSV as pandas export: {path}")


def shift_dataframe_csv(path: Path, dst: Path, offset: int, output_rows: Optional[int] = None) -> None:
    df, _header = _try_read_csv(path)
    shifted = shift_dataframe(df, offset, output_rows=output_rows)
    shifted.to_csv(dst)


def shift_frame_list_file(path: Path, dst: Path, offset: int, frame_count: int) -> None:
    values: list[int] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            values.append(int(stripped))
        except Exception:
            raise RuntimeError(f"Non-integer line encountered in frame-list file: {path}")
    shifted = sorted({val - int(offset) for val in values if 0 <= (val - int(offset)) < frame_count})
    dst.write_text("\n".join(str(v) for v in shifted) + ("\n" if shifted else ""), encoding="utf-8")


def _frame_groups(arr: np.ndarray, frame_col: int = 0) -> dict[int, np.ndarray]:
    if arr.size == 0:
        return {}
    order = np.argsort(arr[:, frame_col], kind="stable")
    ordered = arr[order]
    frames = ordered[:, frame_col].astype(np.int64)
    uniq, starts = np.unique(frames, return_index=True)
    out: dict[int, np.ndarray] = {}
    for idx, frame_num in enumerate(uniq.tolist()):
        start = int(starts[idx])
        end = int(starts[idx + 1]) if idx + 1 < len(starts) else ordered.shape[0]
        out[int(frame_num)] = ordered[start:end]
    return out


def shift_scorer_output_npy(path: Path, dst: Path, offset: int, frame_count: int) -> None:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise RuntimeError(f"Unexpected scorer output format in {path}")
    groups = _frame_groups(arr)
    if len(groups) == 0:
        np.save(dst, arr)
        return

    template = next(iter(groups.values())).copy()
    part_template = template[:, 1].copy()
    for frame_num, rows in groups.items():
        if rows.shape != template.shape or not np.array_equal(rows[:, 1], part_template):
            raise RuntimeError(f"Inconsistent per-frame scorer output layout in {path} at frame {frame_num}")

    empty = template.copy()
    empty[:, 2] = 0
    empty[:, 3] = 0
    empty[:, 4] = 0

    out_rows: list[np.ndarray] = []
    for frame_num in range(frame_count):
        src = frame_num + int(offset)
        rows = groups.get(src)
        block = (rows if rows is not None else empty).copy()
        block[:, 0] = frame_num
        out_rows.append(block)
    np.save(dst, np.vstack(out_rows).astype(arr.dtype, copy=False))


def shift_detected_markers_npy(path: Path, dst: Path, offset: int, frame_count: int, secondary_cam_value: int) -> None:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise RuntimeError(f"Unexpected detected-markers format in {path}")
    target_mask = arr[:, 1].astype(np.int64) == int(secondary_cam_value)
    other_rows = arr[~target_mask].copy()
    target_rows = arr[target_mask]
    groups = _frame_groups(target_rows)

    shifted_rows: list[np.ndarray] = []
    for frame_num in range(frame_count):
        src = frame_num + int(offset)
        rows = groups.get(src)
        if rows is None:
            continue
        block = rows.copy()
        block[:, 0] = frame_num
        block[:, 1] = int(secondary_cam_value)
        shifted_rows.append(block)

    if len(shifted_rows) > 0:
        shifted = np.vstack(shifted_rows)
        out = np.vstack((other_rows, shifted)) if other_rows.size > 0 else shifted
        order = np.lexsort((out[:, 2], out[:, 1], out[:, 0]))
        out = out[order]
    else:
        out = other_rows
    np.save(dst, out.astype(arr.dtype, copy=False))


def shift_timeseries_npy(path: Path, dst: Path, offset: int, output_rows: Optional[int] = None) -> None:
    arr = np.load(path)
    if arr.ndim == 0:
        raise RuntimeError(f"Scalar NPY file is not a supported timeseries transform: {path}")
    if arr.shape[0] == 0:
        np.save(dst, arr)
        return
    target_rows = arr.shape[0] if output_rows is None else int(output_rows)
    if target_rows < 0:
        raise RuntimeError(f"Negative output row count requested for {path}: {target_rows}")
    out = np.empty((target_rows, *arr.shape[1:]), dtype=arr.dtype)
    if np.issubdtype(arr.dtype, np.floating):
        out[:] = np.nan
    elif np.issubdtype(arr.dtype, np.integer):
        out[:] = -1
    elif np.issubdtype(arr.dtype, np.bool_):
        out[:] = False
    else:
        out[:] = 0
    for row in range(target_rows):
        src = row + int(offset)
        if 0 <= src < arr.shape[0]:
            out[row] = arr[src]
    np.save(dst, out)


def verify_transform(kind: TransformKind, original: Path, candidate: Path, expected_rows: Optional[int] = None) -> None:
    if kind == TransformKind.VIDEO:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for video verification.")
        cap_old = cv2.VideoCapture(str(original))
        cap_new = cv2.VideoCapture(str(candidate))
        try:
            old_count = int(cap_old.get(cv2.CAP_PROP_FRAME_COUNT))
            new_count = int(cap_new.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap_old.release()
            cap_new.release()
        wanted = old_count if expected_rows is None else int(expected_rows)
        if new_count != wanted:
            raise RuntimeError(f"Video frame-count mismatch after rewrite: {new_count} vs {wanted}")
        return
    if kind == TransformKind.DATAFRAME_H5:
        df, _key = _open_hdf(candidate)
        if expected_rows is not None and df.shape[0] != expected_rows:
            raise RuntimeError(f"H5 row-count mismatch after rewrite: {df.shape[0]} vs {expected_rows}")
        return
    if kind == TransformKind.DATAFRAME_CSV:
        df, _header = _try_read_csv(candidate)
        if expected_rows is not None and df.shape[0] != expected_rows:
            raise RuntimeError(f"CSV row-count mismatch after rewrite: {df.shape[0]} vs {expected_rows}")
        return
    if kind == TransformKind.FRAME_LIST:
        return
    if kind == TransformKind.SCORER_OUTPUT_NPY:
        arr = np.load(candidate)
        if expected_rows is not None and arr.ndim == 2 and arr.shape[0] > 0:
            rows_per_frame = int(arr.shape[0] / expected_rows) if expected_rows > 0 else 0
            if rows_per_frame * expected_rows != arr.shape[0]:
                raise RuntimeError(f"Scorer output row-count mismatch after rewrite: {arr.shape[0]} vs frame count {expected_rows}")
        return
    if kind == TransformKind.DETECTED_MARKERS_NPY:
        _ = np.load(candidate)
        return
    if kind == TransformKind.NPY_TIMESERIES:
        arr = np.load(candidate)
        if expected_rows is not None and arr.shape[0] != expected_rows:
            raise RuntimeError(f"NPY timeseries row-count mismatch after rewrite: {arr.shape[0]} vs {expected_rows}")
        return
    raise RuntimeError(f"Unhandled transform kind verification: {kind}")


def atomic_replace(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
