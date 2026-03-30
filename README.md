# cam-align

Portable standalone application for manual camera-alignment and offset compensation.

## Goals

- inspect a single session/raw folder
- determine master and secondary cameras
- preview master, raw secondary, and compensated secondary frames side-by-side
- dry-run compensation policy
- run a post-processing check that validates dense paired side/front or left/right outputs
- mutate files in place only after automatic backup
- support explicit undo
- restore originals automatically on failure

## Run

```powershell
python -m pip install -r requirements.txt
python run_app.py
```

Or install editable and run:

```powershell
python -m pip install -e .
cam-align
```

## Notes

- The tool is self-contained and does not import `reachx`.
- It is intended to be portable: copy this folder elsewhere and run it with its own environment.
- Compensation keeps the master timeline authoritative and applies offsets only to selected secondary cameras.
- Compensation uses `systemdata_copy.yaml` to choose the master camera, ignores `cam3`/`stimCam`, shifts the other camera first, and then normalizes only that secondary tail to the master frame count; when a secondary timestamps file is present, it is rewritten with the same shift/length policy so downstream dropped-frame detection remains consistent.
- Timestamp files are also inspected for in-recording dropped frames. If drops are detected, the app warns the user because this tool does not fix mid-recording hardware drops; if no drops are detected, compensation proceeds normally.
- The `Post-Process Check` button uses `ffprobe` for video frame counts and OpenCV frame reads to validate videos and dense paired artifacts after compensation.
- If the master/secondary frame-count mismatch exceeds 100 frames, the tool errors out and treats it as a significant acquisition alignment error.
