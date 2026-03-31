from __future__ import annotations

import json
import subprocess
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional, TypeVar

import numpy as np
import pandas as pd

try:
    import cv2
except Exception:
    cv2 = None

from cam_align_tool.core.models import TransformKind

T = TypeVar("T")


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


def _resolve_ffmpeg_path() -> Optional[str]:
    path = shutil.which("ffmpeg")
    if path:
        return path
    ffprobe = _resolve_ffprobe_path()
    if ffprobe:
        candidate = Path(ffprobe).with_name("ffmpeg.exe")
        if candidate.is_file():
            return str(candidate)
    candidates = [
        Path.home() / "AppData/Local/Microsoft/WinGet/Links/ffmpeg.exe",
        Path("C:/Program Files/CBD/CHITUBOX_Basic/Resources/DependentSoftware/recordOrShot/ffmpeg.exe"),
    ]
    winget_root = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    candidates.extend(sorted(winget_root.glob("Gyan.FFmpeg_*/*/bin/ffmpeg.exe")))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _ffprobe_stream_metadata(path: Path) -> dict[str, str]:
    ffprobe = _resolve_ffprobe_path()
    if ffprobe is None:
        return {}
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,codec_tag_string,avg_frame_rate,width,height,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(out.stdout or "{}")
    except Exception:
        return {}
    streams = payload.get("streams") or []
    if not streams:
        return {}
    stream = streams[0]
    return {str(k): str(v) for k, v in stream.items() if v is not None}


def _write_video_ffmpeg(dst: Path, shifted: list[np.ndarray], width: int, height: int, fps_expr: str) -> bool:
    ffmpeg = _resolve_ffmpeg_path()
    if ffmpeg is None or len(shifted) == 0:
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-nostats",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        fps_expr,
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except Exception:
        return False
    try:
        assert proc.stdin is not None
        for frame in shifted:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        _stdout, stderr = proc.communicate()
    except Exception:
        proc.kill()
        proc.wait()
        return False
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg video rewrite failed for {dst.name}: {stderr.decode('utf-8', errors='replace')[:500]}")
    return True


def _fps_from_expr(fps_expr: str) -> float:
    text = str(fps_expr or "").strip()
    if not text or text in {"0/0", "N/A"}:
        return 0.0
    try:
        return float(Fraction(text))
    except Exception:
        try:
            return float(text)
        except Exception:
            return 0.0


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
        source_meta = _ffprobe_stream_metadata(src)
        fps_expr = source_meta.get("avg_frame_rate", "") or f"{fps:.9f}"
        if fps_expr in {"0/0", "N/A"}:
            fps_expr = f"{fps:.9f}"
        fps_value = _fps_from_expr(fps_expr)
        if fps_value <= 0:
            raise RuntimeError(f"Unable to determine frame rate for video rewrite: {src}")

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

        if dst.suffix.lower() == ".mp4" and _write_video_ffmpeg(dst, shifted, width, height, fps_expr):
            return

        fourcc_tags = ["avc1", "mp4v"] if dst.suffix.lower() == ".mp4" else ["XVID", "MJPG"]
        writer = None
        for tag in fourcc_tags:
            writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*tag), fps_value, (width, height))
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


def _shift_dense_rows(rows: list[T], offset: int, output_count: Optional[int], pad_row: T) -> list[T]:
    target_count = len(rows) if output_count is None else int(output_count)
    if target_count < 0:
        raise RuntimeError(f"Negative output row count requested: {target_count}")
    out: list[T] = []
    for idx in range(target_count):
        src_idx = idx + int(offset)
        if 0 <= src_idx < len(rows):
            out.append(rows[src_idx])
        else:
            out.append(pad_row)
    return out


def _split_timestamp_line(line: str) -> tuple[list[str], str]:
    if "," in line:
        return [part.strip() for part in line.split(",")], ","
    return line.split(), " "


def shift_timestamp_file(path: Path, dst: Path, offset: int, output_rows: Optional[int] = None) -> None:
    raw_lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not raw_lines:
        dst.write_text("", encoding="utf-8")
        return

    parsed = [_split_timestamp_line(line) for line in raw_lines]
    cols = [parts for parts, _sep in parsed]
    sep = parsed[0][1]
    widths = {len(parts) for parts in cols}
    if len(widths) != 1:
        raise RuntimeError(f"Inconsistent timestamp row widths in {path}")

    intervals: list[int] = []
    for parts in cols:
        try:
            intervals.append(int(float(parts[0])))
        except Exception as exc:
            raise RuntimeError(f"Invalid timestamp interval in {path}: {parts[0]}") from exc
    nominal = int(np.median(np.asarray(intervals, dtype=np.int64)))
    if nominal <= 0:
        raise RuntimeError(f"Unable to infer nominal timestamp interval from {path}")

    pad_parts = list(cols[-1])
    pad_parts[0] = str(nominal)
    shifted = _shift_dense_rows(cols, offset, output_rows, pad_parts)
    text = "\n".join(sep.join(parts) for parts in shifted)
    if text:
        text += "\n"
    dst.write_text(text, encoding="utf-8")


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
    if target_rows <= df.shape[0]:
        out = df.iloc[:target_rows].copy(deep=True)
    else:
        out = df.reindex(range(target_rows)).copy(deep=True)
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
        original_meta = _ffprobe_stream_metadata(original)
        candidate_meta = _ffprobe_stream_metadata(candidate)
        original_codec = original_meta.get("codec_name", "")
        candidate_codec = candidate_meta.get("codec_name", "")
        if original_codec == "h264" and candidate_codec and candidate_codec != "h264":
            raise RuntimeError(
                f"Video codec mismatch after rewrite: expected h264-compatible output, got {candidate_codec or 'unknown'}"
            )
        original_rate = original_meta.get("avg_frame_rate", "")
        candidate_rate = candidate_meta.get("avg_frame_rate", "")
        if original_rate and candidate_rate and original_rate not in {"0/0", "N/A"} and candidate_rate not in {"0/0", "N/A"}:
            try:
                old_fps = float(Fraction(original_rate))
                new_fps = float(Fraction(candidate_rate))
            except Exception:
                old_fps = None
                new_fps = None
            if old_fps is not None and new_fps is not None and abs(float(old_fps) - float(new_fps)) > 1e-4:
                raise RuntimeError(
                    f"Video frame-rate mismatch after rewrite: {candidate_rate} vs original {original_rate}"
                )
        return
    if kind == TransformKind.TIMESTAMPS:
        lines = [line for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        if expected_rows is not None and len(lines) != expected_rows:
            raise RuntimeError(f"Timestamp row-count mismatch after rewrite: {len(lines)} vs {expected_rows}")
        if lines:
            parts, _sep = _split_timestamp_line(lines[0])
            try:
                int(float(parts[0]))
            except Exception as exc:
                raise RuntimeError(f"Timestamp file is not parseable after rewrite: {candidate}") from exc
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
