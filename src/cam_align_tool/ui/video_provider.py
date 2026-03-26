from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtGui import QImage

try:
    import cv2
except Exception:
    cv2 = None


class VideoProvider:
    def __init__(self) -> None:
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._paths: dict[str, Path] = {}

    def clear(self) -> None:
        for cap in self._caps.values():
            try:
                cap.release()
            except Exception:
                pass
        self._caps.clear()
        self._paths.clear()

    def set_paths(self, paths: dict[str, Path]) -> None:
        if cv2 is None:
            self.clear()
            return
        self.clear()
        self._paths = {k: v for k, v in paths.items() if v.is_file()}
        for name, path in self._paths.items():
            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                self._caps[name] = cap

    def read_frame(self, camera: str, frame_index: int) -> Optional[QImage]:
        if cv2 is None:
            return None
        cap = self._caps.get(camera)
        if cap is None:
            return None
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            return None
        frame_index = max(0, min(int(frame_index), count - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
