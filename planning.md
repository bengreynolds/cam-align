# Camera Alignment Planning

## Objective

Build a manual camera-alignment tool in two forms:

1. Standalone removable UI:
   - accept raw acquisition/session data input
   - allow user to determine frame offset manually
   - run explicit compensation on videos and relevant data products
   - live under `temp` and remain portable outside this repo
   - not depend on `reachx` runtime modules
2. Integrated ReachX workflow:
   - reuse existing camera views where practical
   - visualize master vs secondary offset
   - let user pick/frame-scrub an offset
   - run explicit compensation over affected files

Implementation is intentionally blocked until plan approval.

Current execution scope after approval:

- Active: standalone only
- Deferred: any ReachX integration until a later explicit request

## Context Summary

### Master timeline behavior already present

- `reachx/data/sessionmgr.py`
  - determines master camera from `*_systemdata_copy.yaml`
  - loads master first
  - pads secondary streams at the front if they started late
  - trims/pads secondary frame counts to match master
- `SessionEventList` corrects event frames for dropped frames on the master only
- `MainCamView` and `CamFrame` assume one canonical frame index across all displayed cameras

### Current standalone base worth reusing

- `temp/analysis-standalone`
  - already has session discovery
  - dual-camera preview
  - frame slider/playback
  - mode split for `legacy` vs `fixed-cam`
  - file loaders for curation, trajectories, auto-segments, and videos

## Critical Design Constraints

1. Master timeline is canonical.
2. Manual alignment offsets must not alter master-camera event timing.
3. Manual offsets must be additive to, not conflated with, dropped-frame correction.
4. Compensation must treat legacy and fixed-cam file schemas separately.
5. The user must explicitly trigger mutation; no silent writeback.
6. The tool should produce a manifest/log of what changed.
7. Mutation is in-place only after an automatic backup is created.
8. Undo must restore original files from backup.
9. Failure handling must restore originals or finish recovery automatically.

## File Classes To Consider

### Primary sources

- master video: reference only, should not be shifted
- secondary videos: candidates for compensation
- per-camera timestamps:
  - `*_timestamps.txt`
  - may need truncation/padding only if the compensated dataset promises file-level consistency

### Session event files

- `*_events.txt`
- `*_events_shifted.txt`

These are master-timeline artifacts and should usually remain unchanged if only secondary cameras are being aligned.

## Safety and Recovery Requirements

### Backup model

Before any in-place mutation:

1. create a transaction folder under the session or chosen work root
2. copy originals for all files that will be mutated
3. write a manifest describing:
   - transaction id
   - source session path
   - target cameras
   - offset values
   - files scheduled for mutation
   - backup file locations
   - derived files scheduled for regeneration/invalidation

### Undo model

- expose an explicit undo action that restores originals from the latest successful backup set
- keep enough metadata to restore filenames and remove newly generated replacements

### Failure recovery model

- use staged writes where possible
- do not overwrite originals until replacement artifacts pass basic validation
- if a write/rebuild step fails:
  - restore backed-up originals automatically
  - remove partial outputs
  - emit failure manifest/status with exact recovery result

### Recreation requirement

- if the policy for a derived file is "regenerate", regeneration should happen automatically as part of the run when feasible
- if automatic regeneration is not feasible for a given artifact in v1, the tool must:
  - restore originals on failure
  - clearly mark that artifact as invalidated or pending regeneration

### Legacy derived artifacts

- `*_filt_data.h5`
- `*_Ordered_Reach_Events.txt`
- per-camera DLC `.h5/.csv`

### Fixed-cam derived artifacts

- `*_left_filtered2D.h5`
- `*_right_filtered2D.h5`
- `*_centered3D.h5`
- `*_eventSegmentation.pickle`
- related pellet/segmentation pickles

### ReachX-native derived artifacts

- scorer folders with:
  - `detected_markers.npy`
  - `detected_reaches.txt`
  - possibly trajectory arrays derived from both cameras

## Major Risk Distinction

Not every file should be "shifted in place."

Safer categories:

- Secondary camera-only 2D per-frame outputs:
  - can usually be padded/truncated/shifted deterministically
- Pure event files on master timeline:
  - usually leave unchanged

Higher-risk categories:

- Multi-camera fused products:
  - `*_filt_data.h5`
  - `*_centered3D.h5`
  - `*_eventSegmentation.pickle`
  - ReachX scorer outputs derived from both views

For those, preferred behavior is likely one of:

1. regenerate from compensated upstream files
2. explicitly invalidate/delete and mark for regeneration
3. support direct rewrite only after schema-specific validation

## Proposed Architecture

### Shared compensation engine

Create a reusable domain layer shared by integrated and standalone UIs:

- session inspection
- master/secondary camera role resolution
- offset preview math
- file classification
- transform planning
- dry-run diff summary
- execution + manifest writing

Suggested responsibilities:

- `session contract inspector`
- `offset application policy`
- `file transformer registry`
- `execution manifest writer`

### Standalone workflow

This must be built first.

Requirements:

- package and run independently from `reachx`
- keep all code self-contained under a dedicated `temp` subtree
- support being copied to another directory and run there with its own dependencies

Standalone v1 workflow:

1. choose mode and input folder
2. auto-detect master/secondary cameras
3. preview synchronized views
4. determine offset by frame scrubbing
5. dry-run summary of files to rewrite/regenerate/invalidate
6. create backup transaction set
7. explicit `Run offset compensation`
8. perform compensation, regeneration, and verification
9. allow explicit undo from backup

### Integrated ReachX workflow

Likely insertion points:

- extension of existing camera/session views rather than a disconnected new viewer
- visualization reuse from `MainCamView`/`CamFrame`
- optional timeline preview tied to current session frame

Integrated v1 workflow:

1. load session
2. open camera alignment tool
3. choose target secondary camera
4. scrub master and secondary frames side-by-side
5. enter or adjust frame offset
6. preview resulting alignment
7. run compensation
8. refresh session state/reload affected data

## Recommended Compensation Semantics

### Offset definition

Use one explicit definition everywhere:

- positive offset = secondary camera content appears late relative to master and must be shifted earlier
- negative offset = secondary camera content appears early relative to master and must be shifted later

This needs to be locked in early to avoid UI/export ambiguity.

### Data transform behavior

For per-frame secondary-camera arrays:

- shift indices
- pad vacated frames with:
  - `NaN`
  - sentinel confidence values
  - duplicated edge frames
  - or explicit "missing" rows

Choice depends on file type.

For secondary video files:

- likely rewrite video with front/back trim or blank/held-frame padding
- avoid changing master video

For fused outputs:

- default to regenerate or invalidate unless direct rewrite is provably correct

### Recommended v1 policy split for high-risk derived artifacts

Option A: Direct rewrite

- Pros:
  - one-click complete output set
  - less follow-up work for user
- Cons:
  - higher schema risk
  - greater chance of silent corruption in fused products
  - more implementation and validation burden in v1

Option B: Invalidate/delete and regenerate automatically when supported

- Pros:
  - safer
  - clearer provenance
  - better fit for fused multi-camera products
- Cons:
  - may take longer
  - regeneration may require more dependencies or source artifacts

Recommended v1:

- Directly rewrite clearly secondary-camera-only artifacts.
- Automatically regenerate fused outputs when the regeneration path is well understood and available in the standalone tool.
- Otherwise preserve originals in backup, mark fused outputs as pending regeneration, and do not claim they were safely rewritten.

## Additional Considerations You Hadn't Explicitly Called Out

1. Offset sign convention must be fixed and shown in UI.
2. Multiple secondary cameras may each need independent offsets.
3. Offsets may differ by session, not rig.
4. Some sessions may already have late-start padding from current loader logic; tool must not double-apply blindly.
5. Dropped-frame artifacts and manual offsets must both be visible in preview.
6. Re-encoding videos can introduce compression drift or generation loss.
7. Existing curated reach files may still be valid if master timeline is untouched, but secondary-derived overlays may not.
8. If raw folder already contains inference/tracking outputs, users need policy choices:
   - rewrite
   - invalidate/delete
   - leave untouched with warning
9. Long-running rewrites need progress, cancel behavior, and crash-safe partial-output handling.
10. The tool should record provenance:
    - who ran it
    - when
    - original offset
    - target cameras
    - files changed
11. In-place mutation without backup is risky for scientific data reproducibility.
12. Need a post-run verification check that master and secondary lengths/contracts still match.

## Suggested v1 Scope

### Included

- manual offset selection by frame scrubbing
- master vs one selected secondary camera at a time
- dry-run summary
- manifest output
- automatic backup transaction creation
- automatic rollback on failure
- explicit undo operation
- compensation of clearly secondary-scoped artifacts
- explicit handling policy for high-risk derived artifacts

### Deferred unless you want it in v1

- automatic offset estimation
- batch compensation across many sessions
- multi-secondary simultaneous solve
- direct patching of every fused downstream file schema
- full parity between standalone and integrated UI before standalone validation

## Execution Plan After Approval

1. Define canonical offset semantics and file policy matrix.
2. Define backup, undo, rollback, and regeneration transaction model.
3. Build standalone session-inspection and transform-planning layer under `temp` with no `reachx` dependency.
4. Implement standalone alignment UI first and validate workflows there.
5. Add manifest generation, dry-run, backup, rollback, undo, and verification.
6. Test against:
   - legacy session with existing `_filt_data.h5`
   - fixed-cam session with existing `*_centered3D.h5` and `*_eventSegmentation.pickle`
7. After standalone validation, integrate with ReachX using existing camera views.

## Clarifying Questions To Resolve Before Implementation

1. Should compensation write into a new sibling output folder by default, or mutate the selected session/raw folder in place?
2. For high-risk fused outputs like `*_filt_data.h5`, `*_centered3D.h5`, and segmentation products, do you want:
   - direct rewrite when possible, or
   - deletion/invalidation plus explicit regeneration guidance?
3. Should v1 target:
   - compressed session folders,
   - raw acquisition folders,
   - or both from the start?
4. Do you want the integrated ReachX version to live as:
   - a separate dock/tool window, or
   - an extension of the main camera/session views?
5. If a session has more than one secondary camera, should v1 support compensating one secondary at a time, or all secondaries in one run?

## Approval Gate

No implementation work should start until you explicitly approve the plan and answer the open policy questions above.
