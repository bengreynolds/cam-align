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
from cam_align_tool.core.regenerate import regenerate_hand_pellet_for_scorer
from cam_align_tool.core.transforms import (
    atomic_replace,
    shift_detected_markers_npy,
    shift_dataframe_csv,
    shift_dataframe_h5,
    shift_frame_list_file,
    shift_scorer_output_npy,
    shift_timestamp_file,
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
_MAX_AUTO_TRIM_FRAMES = 100


def _norm(path: Path) -> str:
    return str(path.resolve()).lower()


def _contains_camera_token(path: Path, camera: str) -> bool:
    return camera.lower() in path.name.lower()


def _supported_scorer_for_path(inspection: InspectionResult, path: Path):
    parent = path.parent.resolve()
    for scorer in inspection.scorer_folders:
        if scorer.supports_hand_pellet_regen and scorer.path.resolve() == parent:
            return scorer
    return None


def _target_frame_count(inspection: InspectionResult, secondary_camera: str) -> int:
    master_rows = inspection.camera_files[inspection.master_camera].frame_count
    secondary_rows = inspection.camera_files[secondary_camera].frame_count
    diff = abs(master_rows - secondary_rows)
    if diff > _MAX_AUTO_TRIM_FRAMES:
        raise RuntimeError(
            "Significant acquisition alignment error: "
            f"{inspection.master_camera}={master_rows}, {secondary_camera}={secondary_rows}, "
            f"diff={diff} frames exceeds allowed limit of {_MAX_AUTO_TRIM_FRAMES}. "
            "Compensation stops instead of trimming/buffering a mismatch this large."
        )
    return master_rows


def _artifact_for_path(path: Path, inspection: InspectionResult, secondary_camera: str, target_frame_count: int) -> Artifact:
    name_lower = path.name.lower()
    master_info = inspection.camera_files.get(inspection.master_camera)
    secondary_info = inspection.camera_files.get(secondary_camera)
    secondary_rows = secondary_info.frame_count if secondary_info is not None else None
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
            row_count=target_frame_count,
        )

    if secondary_info is not None and secondary_info.timestamp_path is not None and _norm(path) == _norm(secondary_info.timestamp_path):
        return Artifact(
            path=path,
            action=ChangeAction.REWRITE,
            reason="secondary_timestamps",
            kind=TransformKind.TIMESTAMPS,
            camera=secondary_camera,
            row_count=target_frame_count,
        )

    if name_lower == f"{secondary_camera.lower()}.npy":
        return Artifact(
            path=path,
            action=ChangeAction.REWRITE,
            reason="secondary_inference_npy",
            kind=TransformKind.SCORER_OUTPUT_NPY,
            camera=secondary_camera,
            row_count=target_frame_count,
        )

    if name_lower == "detected_markers.npy":
        return Artifact(
            path=path,
            action=ChangeAction.REWRITE,
            reason="secondary_rows_in_detected_markers",
            kind=TransformKind.DETECTED_MARKERS_NPY,
            camera=secondary_camera,
            row_count=target_frame_count,
        )

    if name_lower in {"hand.npy", "pellet.npy"}:
        scorer = _supported_scorer_for_path(inspection, path)
        if scorer is not None:
            return Artifact(path=path, action=ChangeAction.INVALIDATE, reason="derived_trajectory_requires_regeneration")
        return Artifact(path=path, action=ChangeAction.SKIP, reason="trajectory_regeneration_not_supported")

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
                row_count=target_frame_count,
            )
        return Artifact(path=path, action=ChangeAction.SKIP, reason="not_target_camera")

    if name_lower.endswith("_processed_frames.txt"):
        return Artifact(
            path=path,
            action=ChangeAction.REWRITE,
            reason="camera_frame_list",
            kind=TransformKind.FRAME_LIST,
            camera=secondary_camera,
            row_count=target_frame_count,
        )

    if path.suffix.lower() == ".h5":
        try:
            import pandas as pd

            with pd.HDFStore(str(path), mode="r") as store:
                keys = list(store.keys())
                row_count = int(store[keys[0]].shape[0]) if keys else None
        except Exception:
            return Artifact(path=path, action=ChangeAction.SKIP, reason="unsupported_h5")
        if secondary_rows is not None and row_count != secondary_rows:
            return Artifact(path=path, action=ChangeAction.SKIP, reason="unexpected_h5_row_count")
        return Artifact(
            path=path,
            action=ChangeAction.REWRITE,
            reason="camera_dataframe_h5",
            kind=TransformKind.DATAFRAME_H5,
            camera=secondary_camera,
            row_count=target_frame_count,
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
        if secondary_rows is not None and row_count != secondary_rows:
            return Artifact(path=path, action=ChangeAction.SKIP, reason="unexpected_csv_row_count")
        return Artifact(
            path=path,
            action=ChangeAction.REWRITE,
            reason="camera_dataframe_csv",
            kind=TransformKind.DATAFRAME_CSV,
            camera=secondary_camera,
            row_count=target_frame_count,
        )

    return Artifact(path=path, action=ChangeAction.SKIP, reason="unsupported_or_unchanged")


def build_plan(inspection: InspectionResult, secondary_camera: str, offset: int) -> CompensationPlan:
    if secondary_camera == inspection.master_camera:
        raise RuntimeError("Manual offset compensation may only target a secondary camera.")
    if secondary_camera not in inspection.camera_files:
        raise RuntimeError(f"Secondary camera not found in session: {secondary_camera}")
    target_frame_count = _target_frame_count(inspection, secondary_camera)

    rewrites: list[PlannedChange] = []
    invalidations: list[PlannedChange] = []
    scorer_regenerations = [scorer for scorer in inspection.scorer_folders if scorer.supports_hand_pellet_regen]
    skips: list[Artifact] = []
    seen: set[str] = set()

    for path in inspection.all_files:
        artifact = _artifact_for_path(path, inspection, secondary_camera, target_frame_count)
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
        target_frame_count=target_frame_count,
        rewrites=len(rewrites),
        invalidations=len(invalidations),
        scorer_regenerations=len(scorer_regenerations),
        skips=len(skips),
    )
    return CompensationPlan(
        inspection=inspection,
        secondary_camera=secondary_camera,
        offset=int(offset),
        target_frame_count=target_frame_count,
        rewrites=rewrites,
        invalidations=invalidations,
        scorer_regenerations=scorer_regenerations,
        skips=skips,
    )


def summarize_plan(plan: CompensationPlan) -> PlanSummary:
    master_rows = plan.inspection.camera_files[plan.inspection.master_camera].frame_count
    secondary_rows = plan.inspection.camera_files[plan.secondary_camera].frame_count
    lines = [
        f"Root: {plan.inspection.root}",
        f"Mode: {plan.inspection.mode.value}",
        f"Master camera: {plan.inspection.master_camera}",
        f"Master frames: {master_rows}",
        f"Secondary target: {plan.secondary_camera}",
        f"Secondary frames: {secondary_rows}",
        f"Target secondary frame count after offset + normalize: {plan.target_frame_count}",
        f"Offset: {plan.offset} (new[t] = raw[t + offset])",
        "",
        f"Rewrite count: {len(plan.rewrites)}",
    ]
    if plan.inspection.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend([f"  - {warning}" for warning in plan.inspection.warnings])
    if master_rows != secondary_rows:
        direction = "trim secondary tail" if secondary_rows > master_rows else "buffer secondary tail"
        lines.insert(6, f"Post-offset length normalization: {direction} by {abs(master_rows - secondary_rows)} frame(s)")
    for item in plan.rewrites:
        lines.append(f"  REWRITE {item.artifact.kind.value}: {item.artifact.path.relative_to(plan.inspection.root)}")
    lines.append(f"Invalidate count: {len(plan.invalidations)}")
    for item in plan.invalidations:
        lines.append(f"  INVALIDATE: {item.artifact.path.relative_to(plan.inspection.root)}")
    lines.append(f"Regeneration count: {len(plan.scorer_regenerations)}")
    for scorer in plan.scorer_regenerations:
        lines.append(f"  REGENERATE scorer trajectories: {scorer.path.relative_to(plan.inspection.root)}")
    lines.append(f"Skip count: {len(plan.skips)}")
    return PlanSummary(
        rewrite_count=len(plan.rewrites),
        invalidate_count=len(plan.invalidations),
        regeneration_count=len(plan.scorer_regenerations),
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


def _find_scorer(inspection: InspectionResult, scorer_name: str):
    for scorer in inspection.scorer_folders:
        if scorer.name == scorer_name:
            return scorer
    return None


def _planned_backup_relpaths(plan: CompensationPlan) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for item in [*plan.rewrites, *plan.invalidations]:
        key = str(item.backup_relpath).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item.backup_relpath)
    return out


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


def execute_plan(plan: CompensationPlan, progress: Optional[Callable[[str], None]] = None,
                 create_backup: bool = True) -> Path:
    root = plan.inspection.root
    txn_dir = _new_transaction_dir(root)
    planned_backup_paths = [str(rel) for rel in _planned_backup_relpaths(plan)] if create_backup else []
    planned_created_paths = []
    for scorer in plan.scorer_regenerations:
        for name in ("hand.npy", "pellet.npy"):
            rel = Path(scorer.path.relative_to(root), name)
            if not (root / rel).is_file():
                planned_created_paths.append(str(rel))
    manifest: dict = {
        "status": "in_progress",
        "created_utc": datetime.utcnow().isoformat(),
        "root": str(root),
        "mode": plan.inspection.mode.value,
        "master_camera": plan.inspection.master_camera,
        "secondary_camera": plan.secondary_camera,
        "offset": plan.offset,
        "target_frame_count": plan.target_frame_count,
        "backup_enabled": bool(create_backup),
        "rewrite_paths": [str(item.artifact.path.relative_to(root)) for item in plan.rewrites],
        "invalidated_paths": [str(item.artifact.path.relative_to(root)) for item in plan.invalidations],
        "scorer_regenerations": [str(scorer.path.relative_to(root)) for scorer in plan.scorer_regenerations],
        "planned_backup_paths": planned_backup_paths,
        "backed_up_paths": [],
        "restored_paths": [],
        "generated_paths": [],
        "created_paths": planned_created_paths,
        "undone": False,
    }
    write_json(_manifest_path(txn_dir), manifest)

    def emit(msg: str) -> None:
        if callable(progress):
            progress(msg)
        log_event(_LOG, "execute_plan_progress", message=msg)

    try:
        backups: dict[str, Path] = {}
        if create_backup:
            for rel in _planned_backup_relpaths(plan):
                backup = _backup_original(root, txn_dir, rel)
                backups[str(rel).lower()] = backup
                manifest["backed_up_paths"].append(str(rel))
                write_json(_manifest_path(txn_dir), manifest)
                emit(f"Backed up {rel}")
            if sorted(manifest["backed_up_paths"]) != sorted(planned_backup_paths):
                raise RuntimeError("Backup verification failed before mutation; not all planned originals were copied.")
        else:
            emit("Backup creation disabled; proceeding without transaction copies.")

        for item in plan.rewrites:
            artifact = item.artifact
            rel = item.backup_relpath
            source_for_verify = backups[str(rel).lower()] if create_backup else artifact.path
            tmp_path = txn_dir / "staging" / rel
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            if artifact.kind == TransformKind.VIDEO:
                shift_video_file(artifact.path, tmp_path, plan.offset, output_frame_count=artifact.row_count)
            elif artifact.kind == TransformKind.TIMESTAMPS:
                shift_timestamp_file(artifact.path, tmp_path, plan.offset, output_rows=artifact.row_count)
            elif artifact.kind == TransformKind.DATAFRAME_H5:
                shift_dataframe_h5(artifact.path, tmp_path, plan.offset, output_rows=artifact.row_count)
            elif artifact.kind == TransformKind.DATAFRAME_CSV:
                shift_dataframe_csv(artifact.path, tmp_path, plan.offset, output_rows=artifact.row_count)
            elif artifact.kind == TransformKind.FRAME_LIST:
                frame_count = plan.target_frame_count
                shift_frame_list_file(artifact.path, tmp_path, plan.offset, frame_count)
            elif artifact.kind == TransformKind.SCORER_OUTPUT_NPY:
                frame_count = plan.target_frame_count
                shift_scorer_output_npy(artifact.path, tmp_path, plan.offset, frame_count)
            elif artifact.kind == TransformKind.DETECTED_MARKERS_NPY:
                frame_count = plan.target_frame_count
                camera_value = plan.inspection.cameras.index(plan.secondary_camera)
                if plan.inspection.mode.value == "legacy":
                    camera_value = {"sideCam": 0, "frontCam": 1, "stimCam": 2, "fastCam": 3}.get(plan.secondary_camera, camera_value)
                else:
                    camera_value = {"left": 4, "right": 5}.get(plan.secondary_camera, camera_value)
                shift_detected_markers_npy(artifact.path, tmp_path, plan.offset, frame_count, camera_value)
            else:
                raise RuntimeError(f"Unhandled rewrite kind: {artifact.kind}")
            verify_transform(artifact.kind, source_for_verify, tmp_path, artifact.row_count)
            atomic_replace(tmp_path, artifact.path)
            emit(f"Rewrote {rel}")

        for item in plan.invalidations:
            rel = item.backup_relpath
            item.artifact.path.unlink(missing_ok=True)
            emit(f"Invalidated {rel}")

        generated_paths: list[str] = []
        for scorer in plan.scorer_regenerations:
            emit(f"Regenerating hand/pellet trajectories for scorer {scorer.name}")
            for out_path in regenerate_hand_pellet_for_scorer(scorer, plan.inspection, progress=progress):
                rel = str(out_path.relative_to(root))
                generated_paths.append(rel)
        manifest["generated_paths"] = generated_paths

        manifest["status"] = "completed"
        write_json(_manifest_path(txn_dir), manifest)
        emit(f"Completed transaction {txn_dir.name}")
        return _manifest_path(txn_dir)
    except Exception as exc:
        if create_backup:
            emit(f"Failure encountered, rolling back: {exc}")
            rollback_transaction(root, txn_dir, manifest=manifest, progress=progress, error=str(exc))
        else:
            emit(f"Failure encountered with backups disabled; rollback unavailable: {exc}")
            staging_root = txn_dir / "staging"
            if staging_root.exists():
                shutil.rmtree(staging_root, ignore_errors=True)
            manifest["status"] = "failed_no_backup"
            manifest["rollback_error"] = str(exc)
            write_json(_manifest_path(txn_dir), manifest)
        raise


def regenerate_scorer_outputs(
    inspection: InspectionResult,
    scorer_name: str,
    progress: Optional[Callable[[str], None]] = None,
    create_backup: bool = True,
) -> Path:
    scorer = _find_scorer(inspection, scorer_name)
    if scorer is None:
        raise RuntimeError(f"Scorer folder not found: {scorer_name}")
    if not scorer.supports_hand_pellet_regen:
        reason = scorer.regen_note or "required scorer files are not available"
        raise RuntimeError(f"Scorer '{scorer.name}' does not support hand/pellet regeneration: {reason}")

    root = inspection.root
    txn_dir = _new_transaction_dir(root)
    output_relpaths = [Path(scorer.path.relative_to(root), name) for name in ("hand.npy", "pellet.npy")]
    existing_backup_paths = [str(rel) for rel in output_relpaths if (root / rel).is_file()] if create_backup else []
    created_paths = [str(rel) for rel in output_relpaths if not (root / rel).is_file()]
    manifest: dict = {
        "status": "in_progress",
        "created_utc": datetime.utcnow().isoformat(),
        "root": str(root),
        "mode": inspection.mode.value,
        "transaction_type": "scorer_hand_pellet_regeneration",
        "scorer_folder": str(scorer.path.relative_to(root)),
        "backup_enabled": bool(create_backup),
        "planned_backup_paths": existing_backup_paths,
        "backed_up_paths": [],
        "generated_paths": [],
        "created_paths": created_paths,
        "restored_paths": [],
        "undone": False,
    }
    write_json(_manifest_path(txn_dir), manifest)

    def emit(msg: str) -> None:
        if callable(progress):
            progress(msg)
        log_event(_LOG, "regenerate_scorer_outputs_progress", message=msg)

    try:
        if create_backup:
            for rel_str in existing_backup_paths:
                rel = Path(rel_str)
                _backup_original(root, txn_dir, rel)
                manifest["backed_up_paths"].append(rel_str)
                write_json(_manifest_path(txn_dir), manifest)
                emit(f"Backed up {rel}")
        else:
            emit("Backup creation disabled; proceeding without transaction copies.")

        generated_paths = regenerate_hand_pellet_for_scorer(scorer, inspection, progress=progress)
        generated_rel = [str(path.relative_to(root)) for path in generated_paths]
        manifest["generated_paths"] = generated_rel
        manifest["status"] = "completed"
        write_json(_manifest_path(txn_dir), manifest)
        emit(f"Completed transaction {txn_dir.name}")
        return _manifest_path(txn_dir)
    except Exception as exc:
        if create_backup:
            emit(f"Failure encountered, rolling back: {exc}")
            rollback_transaction(root, txn_dir, manifest=manifest, progress=progress, error=str(exc))
        else:
            emit(f"Failure encountered with backups disabled; rollback unavailable: {exc}")
            manifest["status"] = "failed_no_backup"
            manifest["rollback_error"] = str(exc)
            write_json(_manifest_path(txn_dir), manifest)
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

    for rel_str in manifest.get("created_paths", []):
        if rel_str in restored:
            continue
        dst = root / rel_str
        if dst.is_file():
            dst.unlink(missing_ok=True)
            emit(f"Removed generated file {rel_str}")

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
    if not manifest.get("backup_enabled", True):
        raise RuntimeError("The most recent completed compensation was run with backup creation disabled, so undo is unavailable.")
    rollback_transaction(root, txn_dir, manifest=manifest, progress=progress, error="undo")
    manifest = json.loads(_manifest_path(txn_dir).read_text(encoding="utf-8"))
    manifest["status"] = "undone"
    manifest["undone"] = True
    write_json(_manifest_path(txn_dir), manifest)
    return _manifest_path(txn_dir)


def inspect_and_plan(root: Path, secondary_camera: str, offset: int) -> CompensationPlan:
    inspection = inspect_input_folder(root)
    return build_plan(inspection, secondary_camera, offset)
