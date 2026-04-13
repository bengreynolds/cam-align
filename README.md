# cam-align

Standalone camera alignment and offset compensation app.

## What It Does

- Opens a single session or raw camera folder
- Identifies the master camera and the secondary camera
- Previews master, raw secondary, and compensated secondary frames side by side
- Supports dry-run compensation
- Runs a post-process check for paired side/front or left/right output
- Backs up files before in-place changes
- Supports explicit undo and failure recovery

## Install And Run

Install dependencies and run the GUI from the repo:

```powershell
python -m pip install -r requirements.txt
python run_app.py
```

Or install the package locally and use the console script:

```powershell
python -m pip install -e .
cam-align
```

## Requirements

- Python 3.10 or newer
- PySide6
- NumPy
- pandas
- OpenCV
- tables
- PyYAML
- SciPy

## Notes

- The tool is self-contained and does not import `reachx`.
- It is intended to be portable. Copy the folder elsewhere and run it in its own environment.
- Compensation keeps the master timeline authoritative and applies offsets only to selected secondary cameras.
- `systemdata_copy.yaml` is used to choose the master camera.
- `cam3` and `stimCam` are ignored by the compensation flow.
- Timestamp files are inspected for dropped frames. If drops are detected, the app warns the user because it does not correct mid-recording hardware drops.
- The `Post-Process Check` button uses `ffprobe` and OpenCV frame reads to verify videos and paired artifacts after compensation.
- If the master/secondary frame count mismatch exceeds 100 frames, the tool stops and treats it as a significant acquisition error.

## Typical Workflow

1. Open a session folder in the app.
2. Confirm the master camera selection.
3. Preview the raw and compensated secondary videos.
4. Run a dry pass if you want to inspect the offset before writing files.
5. Apply compensation after the backup is created.
6. Run the post-process check to verify the output.
