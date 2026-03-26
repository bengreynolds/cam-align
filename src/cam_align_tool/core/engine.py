from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from cam_align_tool.config.logging_utils import log_event
from cam_align_tool.core.inspect import inspect_input_folder
from cam_align_tool.core.models import (
    Artifact,
    ChangeAction,
    CompensationPlan,
    InspectionResult,
    PlanSummary,
    PlannedChange,
    TransformKind,
)
from cam_align_tool.core.transforms import (
    atomic_replace,
    shift_detected_markers_npy,
    shift_dataframe_csv,
    shift_dataframe_h5,
    shift_frame_list_file,
    shift_scorer_output_npy,
    shift_timeseries_npy,
    shift_video_file,
    verify_transform,
    write_json,
)

_LOG = logging.getLogger("cam_align_tool.core.engine")
_BACKUP_DIR_NAME = ".cam_align_backup"

_HIGH_RISK_PATTERNS = [
    "_eventSegmentation.pickle",
    "_pelletHistory.pickle",
    "_Ordered_Reach_Events.txt",
    "detected_reaches.txt",
]

_SHIFTABLE_FUSED_H5_PATTERNS = [
    "_filt_data.h5",
    "_centered3D.h5",
    "_filtered3D.h5",
    "_filtered3d_pixels.h5",
    "_pix3d.h5",
]


def _norm(path: Path) -> str:
    return str(path.resolve()).lower()


def _contains_camera_token(path: Path, camera: str) -> bool:
    return camera.lower() in path.name.lower()


def _artifact_for_path(path: Path, inspection: InspectionResult, secondary_camera: str) -> Artifact:
    name_lower = path.name.lower()
    master_info = inspection.camera_files.get(inspection.master_camera)
    secondary_info = inspection.camera_files.get(secondary_camera)
    expected_rows = secondary_info.frame_count if secondary_info is not None else None
    master_rows = master_info.frame_count if master_info is not None else None
    if any(name_lower.endswith(pattern.lower()) for pattern in _HIGH_RISK_PATTERNS):
        return Artifact(path=path, action=ChangeAction.INVALIDATE, reason="high_risk_fused_output")

    if secondary_info is not None and _norm(path) == _norm(secondary_info.video_path):
        return Artifact(
            path=path,
            action=ChangeAction.REWRITE,
            reason="secondary_video",
            kind=TransformKind.VIDEO,
            camera=secondary_camera,
            row_count=secondary_info.frame_count,
        )

    if name_lower == f"{secondary_camera.lower()}.npy":
        return Artifact(
            path=path,
            action=ChangeAction.REWRITE,
            reason="secondary_inference_npy",
            kind=TransformKind.SCORER_OUTPUT_NPY,
            camera=secondary_camera,
            row_count=master_rows,
        )

    if name_lower == "detected_markers.npy":
        return Artifact(
            path=path,
            action=ChangeAction.REWRITE,
            reason="secondary_rows_in_detected_markers",
            kind=TransformKind.DETECTED_MARKERS_NPY,
            camera=secondary_camera,
            row_count=master_rows,
        )

    if name_lower in {"hand.npy", "pellet.npy"}:
        return Artifact(
            path=path,
            action=ChangeAction.REWRITE,
            reason="derived_timeseries_npy",
            kind=TransformKind.NPY_TIMESERIES,
            row_count=master_rows,
        )

    if not _contains_camera_token(path, secondary_camera):
        if path.suffix.lower() == ".h5" and any(name_lower.endswith(pattern) for pattern in _SHIFTABLE_FUSED_H5_PATTERNS):
            try:
                import pandas as pd

                with pd.HDFStore(str(path), mode="r") as store:
                    keys = list(store.keys())
                    row_count = int(store[keys[0]].shape[0]) if keys else None
            except Exception:
                return Artifact(path=path, action=ChangeAction.SKIP, reason="unsupported_fused_h5")
            if master_rows is not None and row_count != master_rows:
                return Artifact(path=path, action=ChangeAction.SKIP, reason="unexpected_fused_h5_row_count")
            return Artifact(
                path=path,
                action=ChangeAction.REWRITE,
                reason="derived_fused_dataframe_h5",
                kind=TransformKind.DATAFRAME_H5,
                row_count=row_count,
            )
        return Artifact(path=path, action=ChangeAction.SKIP, reason="not_target_camera")

    if name_lower.endswith("_processed_frames.txt"):
        return Artifact(
            path=path,
            action=ChangeAction.REWRITE,
            reason="camera_frame_list",
            kind=TransformKind.FRAME_LIST,
            camera=secondary_camera,
            row_count=expected_rows,
        )

    if path.suffix.lower() == ".h5":
        try:
            import pandas as pd

            with pd.HDFStore(str(path), mode="r") as store:
                keys = list(store.keys())
                row_count = int(store[keys[0]].shape[0]) if keys else None
        except Exception:
            return Artifact(path=path, action=ChangeAction.SKIP, reason="unsupported_h5")
        if expected_rows is not None and row_count != expected_rows:
            return Artifact(path=path, action=ChangeAction.SKIP, reason="unexpected_h5_row_count")
        return Artifact(
            path=path,
            action=ChangeAction.REWRITE,
            reason="camera_dataframe_h5",
            kind=TransformKind.DATAFRAME_H5,
            camera=secondary_camera,
            row_count=row_count,
        )

    if path.suffix.lower() == ".csv":
        try:
            import pandas as pd

            best_rows = None
            for header in ([0, 1, 2], [0, 1], [0]):
                try:
                    rows = int(pd.read_csv(path, header=header, index_col=0).shape[0])
                    if best_rows is None or rows > best_rows:
                        best_rows = rows
                except Exception:
                    continue
            if best_rows is None:
                raise RuntimeError("Unsupported CSV layout")
            row_count = best_rows
        except Exception:
            return Artifact(path=path, action=ChangeAction.SKIP, reason="unsupported_csv")
        if expected_rows is not None and row_count != expected_rows:
            return Artifact(path=path, action=ChangeAction.SKIP, reason="unexpected_csv_row_count")
        return Artifact(
            path=path,
            action=ChangeAction.REWRITE,
            reason="camera_dataframe_csv",
            kind=TransformKind.DATAFRAME_CSV,
            camera=secondary_camera,
            row_count=row_count,
        )

    return Artifact(path=path, action=ChangeAction.SKIP, reason="unsupported_or_unchanged")


def build_plan(inspection: InspectionResult, secondary_camera: str, offset: int) -> CompensationPlan:
    if secondary_camera == inspection.master_camera:
        raise RuntimeError("Manual offset compensation may only target a secondary camera.")
    if secondary_camera not in inspection.camera_files:
        raise RuntimeError(f"Secondary camera not found in session: {secondary_camera}")

    rewrites: list[PlannedChange] = []
    invalidations: list[PlannedChange] = []
    skips: list[Artifact] = []
    seen: set[str] = set()

    for path in inspection.all_files:
        artifact = _artifact_for_path(path, inspection, secondary_camera)
        rel = path.relative_to(inspection.root)
        key = _norm(path)
        if key in seen:
            continue
        seen.add(key)
        if artifact.action == ChangeAction.REWRITE:
            rewrites.append(PlannedChange(artifact=artifact, backup_relpath=rel))
        elif artifact.action == ChangeAction.INVALIDATE:
            invalidations.append(PlannedChange(artifact=artifact, backup_relpath=rel))
        else:
            skips.append(artifact)

    log_event(
        _LOG,
        "build_plan",
        root=inspection.root,
        secondary=secondary_camera,
        offset=offset,
        rewrites=len(rewrites),
        invalidations=len(invalidations),
        skips=len(skips),
    )
    return CompensationPlan(
        inspection=inspection,
        secondary_camera=secondary_camera,
        offset=int(offset),
        rewrites=rewrites,
        invalidations=invalidations,
        skips=skips,
    )


def summarize_plan(plan: CompensationPlan) -> PlanSummary:
    lines = [
        f"Root: {plan.inspection.root}",
        f"Mode: {plan.inspection.mode.value}",
        f"Master camera: {plan.inspection.master_camera}",
        f"Secondary target: {plan.secondary_camera}",
        f"Offset: {plan.offset} (new[t] = raw[t + offset])",
        "",
        f"Rewrite count: {len(plan.rewrites)}",
    ]
    for item in plan.rewrites:
        lines.append(f"  REWRITE {item.artifact.kind.value}: {item.artifact.path.relative_to(plan.inspection.root)}")
    lines.append(f"Invalidate count: {len(plan.invalidations)}")
    for item in plan.invalidations:
        lines.append(f"  INVALIDATE: {item.artifact.path.relative_to(plan.inspection.root)}")
    lines.append(f"Skip count: {len(plan.skips)}")
    return PlanSummary(
        rewrite_count=len(plan.rewrites),
        invalidate_count=len(plan.invalidations),
        skip_count=len(plan.skips),
        details="\n".join(lines),
    )


def _transaction_root(root: Path) -> Path:
    return root / _BACKUP_DIR_NAME


def _new_transaction_dir(root: Path) -> Path:
    txn = datetime.now().strftime("txn-%Y%m%d-%H%M%S")
    out = _transaction_root(root) / txn
    out.mkdir(parents=True, exist_ok=False)
    return out


def _manifest_path(txn_dir: Path) -> Path:
    return txn_dir / "manifest.json"


def _backup_original(root: Path, txn_dir: Path, relpath: Path) -> Path:
    src = root / relpath
    dst = txn_dir / "originals" / relpath
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _find_last_completed_txn(root: Path) -> Optional[tuple[Path, dict]]:
    txn_root = _transaction_root(root)
    if not txn_root.is_dir():
        return None
    for txn_dir in sorted([p for p in txn_root.iterdir() if p.is_dir()], reverse=True):
        manifest_path = _manifest_path(txn_dir)
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("status") == "completed":
            return txn_dir, manifest
    return None


def execute_plan(plan: CompensationPlan, progress: Optional[Callable[[str], None]] = None) -> Path:
    root = plan.inspection.root
    txn_dir = _new_transaction_dir(root)
    manifest: dict = {
        "status": "in_progress",
        "created_utc": datetime.utcnow().isoformat(),
        "root": str(root),
        "mode": plan.inspection.mode.value,
        "master_camera": plan.inspection.master_camera,
        "secondary_camera": plan.secondary_camera,
        "offset": plan.offset,
        "rewrite_paths": [str(item.artifact.path.relative_to(root)) for item in plan.rewrites],
        "invalidated_paths": [str(item.artifact.path.relative_to(root)) for item in plan.invalidations],
        "restored_paths": [],
        "undone": False,
    }
    write_json(_manifest_path(txn_dir), manifest)

    def emit(msg: str) -> None:
        if callable(progress):
            progress(msg)
        log_event(_LOG, "execute_plan_progress", message=msg)

    try:
        for item in plan.rewrites:
            artifact = item.artifact
            rel = item.backup_relpath
            backup = _backup_original(root, txn_dir, rel)
            emit(f"Backed up {rel}")
            tmp_path = txn_dir / "staging" / rel
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            if artifact.kind == TransformKind.VIDEO:
                shift_video_file(artifact.path, tmp_path, plan.offset)
            elif artifact.kind == TransformKind.DATAFRAME_H5:
                shift_dataframe_h5(artifact.path, tmp_path, plan.offset)
            elif artifact.kind == TransformKind.DATAFRAME_CSV:
                shift_dataframe_csv(artifact.path, tmp_path, plan.offset)
            elif artifact.kind == TransformKind.FRAME_LIST:
                frame_count = plan.inspection.camera_files[plan.secondary_camera].frame_count
                shift_frame_list_file(artifact.path, tmp_path, plan.offset, frame_count)
            elif artifact.kind == TransformKind.SCORER_OUTPUT_NPY:
                frame_count = plan.inspection.camera_files[plan.inspection.master_camera].frame_count
                shift_scorer_output_npy(artifact.path, tmp_path, plan.offset, frame_count)
            elif artifact.kind == TransformKind.DETECTED_MARKERS_NPY:
                frame_count = plan.inspection.camera_files[plan.inspection.master_camera].frame_count
                camera_value = plan.inspection.cameras.index(plan.secondary_camera)
                if plan.inspection.mode.value == "legacy":
                    camera_value = {"sideCam": 0, "frontCam": 1, "stimCam": 2, "fastCam": 3}.get(plan.secondary_camera, camera_value)
                else:
                    camera_value = {"left": 4, "right": 5}.get(plan.secondary_camera, camera_value)
                shift_detected_markers_npy(artifact.path, tmp_path, plan.offset, frame_count, camera_value)
            elif artifact.kind == TransformKind.NPY_TIMESERIES:
                shift_timeseries_npy(artifact.path, tmp_path, plan.offset)
            else:
                raise RuntimeError(f"Unhandled rewrite kind: {artifact.kind}")
            verify_transform(artifact.kind, backup, tmp_path, artifact.row_count)
            atomic_replace(tmp_path, artifact.path)
            emit(f"Rewrote {rel}")

        for item in plan.invalidations:
            rel = item.backup_relpath
            _backup_original(root, txn_dir, rel)
            item.artifact.path.unlink(missing_ok=True)
            emit(f"Invalidated {rel}")

        manifest["status"] = "completed"
        write_json(_manifest_path(txn_dir), manifest)
        emit(f"Completed transaction {txn_dir.name}")
        return _manifest_path(txn_dir)
    except Exception as exc:
        emit(f"Failure encountered, rolling back: {exc}")
        rollback_transaction(root, txn_dir, manifest=manifest, progress=progress, error=str(exc))
        raise


def rollback_transaction(root: Path, txn_dir: Path, manifest: Optional[dict] = None,
                         progress: Optional[Callable[[str], None]] = None,
                         error: str = "") -> None:
    manifest = manifest or json.loads(_manifest_path(txn_dir).read_text(encoding="utf-8"))

    def emit(msg: str) -> None:
        if callable(progress):
            progress(msg)
        log_event(_LOG, "rollback_progress", message=msg)

    originals_root = txn_dir / "originals"
    restored: list[str] = []
    for backup in sorted(originals_root.rglob("*")):
        if not backup.is_file():
            continue
        rel = backup.relative_to(originals_root)
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup, dst)
        restored.append(str(rel))
        emit(f"Restored {rel}")

    staging_root = txn_dir / "staging"
    if staging_root.exists():
        shutil.rmtree(staging_root, ignore_errors=True)

    manifest["status"] = "rolled_back"
    manifest["rollback_error"] = error
    manifest["restored_paths"] = restored
    write_json(_manifest_path(txn_dir), manifest)


def undo_last_transaction(root: Path, progress: Optional[Callable[[str], None]] = None) -> Path:
    found = _find_last_completed_txn(root)
    if found is None:
        raise RuntimeError("No completed compensation transaction available to undo.")
    txn_dir, manifest = found
    rollback_transaction(root, txn_dir, manifest=manifest, progress=progress, error="undo")
    manifest = json.loads(_manifest_path(txn_dir).read_text(encoding="utf-8"))
    manifest["status"] = "undone"
    manifest["undone"] = True
    write_json(_manifest_path(txn_dir), manifest)
    return _manifest_path(txn_dir)


def inspect_and_plan(root: Path, secondary_camera: str, offset: int) -> CompensationPlan:
    inspection = inspect_input_folder(root)
    return build_plan(inspection, secondary_camera, offset)
