# Camera Alignment Agent

## Purpose

Own discovery, planning, and implementation for a manual camera-alignment tool that compensates secondary-camera timing offsets while preserving the master-camera timeline as the canonical reference.

## Current Status

- Phase: context built, planning only
- Implementation: not started
- Approval gate: required before code changes outside this planning folder

## Non-Negotiable Behavioral Rules

1. The master/primary camera timeline remains authoritative.
2. Recorded event timing stays anchored to the master timeline.
3. Manual offsets are applied only to secondary cameras.
4. Existing dropped-frame handling must remain distinct from manual offset compensation.
5. Compensation actions must be explicit, previewable, and user-triggered.
6. File mutation must be auditable and reversible via automatic backup + undo support.
7. If compensation fails mid-run, the tool must automatically restore the original state or complete recovery deterministically.
8. The standalone tool is implemented first and must not depend on `reachx` runtime code.
9. The standalone tool must remain portable from `temp` and runnable after being copied elsewhere.

## Repo Context Captured

- Main integrated app:
  - `reachx/data/sessionmgr.py`
  - `reachx/gui/maincamview.py`
  - `reachx/gui/camframe.py`
  - `reachx/gui/sessionview.py`
  - `reachx/gui/analysisview.py`
- Standalone prototype:
  - `temp/analysis-standalone/src/analysis_standalone/ui/main_window.py`
  - `temp/analysis-standalone/src/analysis_standalone/ui/video_provider.py`
  - `temp/analysis-standalone/src/analysis_standalone/data/*.py`
- Legacy/fixed-cam historical pipelines:
  - `temp/reach-training/PythonScripts/findReachEvents_v2.py`
  - `temp/reach-training/PythonScripts/Reach_Curator_py38_v3.py`
  - `temp/reach-training-fixed-cam/PythonScripts/prepare_reach_data.py`
  - `temp/reach-training-fixed-cam/PythonScripts/parse_pellet_presentations.py`

## Key Findings Driving Design

- ReachX already treats the master camera as the session timeline.
- `SessionEventList` corrects events only for dropped frames on the master camera.
- `CamSource` already pads non-master cameras at the beginning when they start late relative to master.
- Legacy and fixed-cam datasets have different derived-file contracts and cannot share one blind rewrite routine.
- Several downstream artifacts are derived from multi-camera data and are safer to regenerate than to patch naively.

## Agent Role In This Chat

1. Build a safe implementation plan before touching production code.
2. Surface edge cases and file-contract risks early.
3. Prefer a shared compensation engine used by both:
   - a standalone portable UI implemented first
   - an integrated ReachX widget/tool implemented second
4. Keep the standalone implementation independent from the current repo's runtime modules.
5. Delay implementation until plan approval is explicit.

## Expected Deliverables After Approval

1. Shared offset-compensation domain layer and file-transform engine.
2. Standalone manual alignment application/workflow under `temp`, portable outside the repo.
3. Integrated ReachX manual alignment widget/workflow using existing camera views.
4. Logging, dry-run preview, manifest output, backup/undo, rollback-on-failure, and verification path.
