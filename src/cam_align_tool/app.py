from __future__ import annotations

import logging
import os
import sys

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from cam_align_tool.config.logging_utils import configure_logging, log_event
from cam_align_tool.config.settings import load_settings


def main() -> int:
    configure_logging(level=logging.INFO)
    log = logging.getLogger("cam_align_tool.app")
    log_event(log, "app_start", python=sys.version.split()[0], platform=sys.platform)

    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:
        print(f"PySide6 import failed: {exc}", file=sys.stderr)
        return 2

    try:
        from cam_align_tool.ui.main_window import MainWindow
    except Exception as exc:
        print(
            "Application bootstrap failed. Ensure required packages are installed, "
            "including opencv-python and PyYAML.\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 2

    app = QApplication(sys.argv)
    if sys.platform.startswith("win"):
        app.setStyle("Fusion")
    win = MainWindow(settings=load_settings())
    win.show()
    code = app.exec()
    log_event(log, "app_exit", exit_code=code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
