from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QComboBox,
    QVBoxLayout,
    QWidget,
)

from cam_align_tool.config.logging_utils import log_event
from cam_align_tool.config.settings import AppSettings, save_settings
from cam_align_tool.core.engine import execute_plan, inspect_and_plan, summarize_plan, undo_last_transaction
from cam_align_tool.core.inspect import inspect_input_folder
from cam_align_tool.core.models import CompensationPlan, InspectionResult
from cam_align_tool.ui.video_provider import VideoProvider

_LOG = logging.getLogger("cam_align_tool.ui.main_window")


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self._settings = settings
        self._inspection: Optional[InspectionResult] = None
        self._plan: Optional[CompensationPlan] = None
        self._video_provider = VideoProvider()

        self.setWindowTitle("cam-align")
        self.resize(1680, 920)
        self._build_ui()
        self.root_edit.setText(settings.last_root)
        self.offset_spin.setValue(settings.last_offset)
        self._set_status("Select a session/raw folder and inspect it.")

    def _build_ui(self) -> None:
        central = QWidget()
        outer = QVBoxLayout()

        self.root_edit = QLineEdit()
        self.root_browse_btn = QPushButton("Browse...")
        self.inspect_btn = QPushButton("Inspect Folder")
        self.master_label = QLabel("-")
        self.mode_label = QLabel("-")
        self.secondary_combo = QComboBox()
        self.offset_spin = QSpinBox()
        self.offset_spin.setRange(-5000, 5000)
        self.offset_spin.setSingleStep(1)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_label = QLabel("Frame: 0")
        self.raw_index_label = QLabel("Raw secondary idx: -")
        self.preview_index_label = QLabel("Preview secondary idx: -")
        self.sign_label = QLabel("Offset convention: compensated[t] = raw[t + offset]")
        self.sign_label.setStyleSheet("color: #b71c1c; font-weight: 600;")

        self.dry_run_btn = QPushButton("Dry Run")
        self.run_btn = QPushButton("Run Offset Compensation")
        self.undo_btn = QPushButton("Undo Last Compensation")

        top_form = QFormLayout()
        root_row = QWidget()
        root_row_layout = QHBoxLayout()
        root_row_layout.setContentsMargins(0, 0, 0, 0)
        root_row_layout.addWidget(self.root_edit, 1)
        root_row_layout.addWidget(self.root_browse_btn)
        root_row_layout.addWidget(self.inspect_btn)
        root_row.setLayout(root_row_layout)
        top_form.addRow("Input folder:", root_row)
        top_form.addRow("Mode:", self.mode_label)
        top_form.addRow("Master camera:", self.master_label)
        top_form.addRow("Secondary target:", self.secondary_combo)
        top_form.addRow("Offset:", self.offset_spin)
        top_form.addRow("Sign convention:", self.sign_label)

        frame_row = QWidget()
        frame_row_layout = QHBoxLayout()
        frame_row_layout.setContentsMargins(0, 0, 0, 0)
        frame_row_layout.addWidget(self.frame_slider, 1)
        frame_row_layout.addWidget(self.frame_label)
        frame_row.setLayout(frame_row_layout)
        top_form.addRow("Master frame:", frame_row)
        top_form.addRow("", self.raw_index_label)
        top_form.addRow("", self.preview_index_label)

        actions = QWidget()
        actions_layout = QHBoxLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.addWidget(self.dry_run_btn)
        actions_layout.addWidget(self.run_btn)
        actions_layout.addWidget(self.undo_btn)
        actions_layout.addStretch(1)
        actions.setLayout(actions_layout)
        top_form.addRow("Actions:", actions)

        form_host = QWidget()
        form_host.setLayout(top_form)
        outer.addWidget(form_host)

        grid = QGridLayout()
        self.master_name = QLabel("Master")
        self.secondary_raw_name = QLabel("Secondary Raw")
        self.secondary_preview_name = QLabel("Secondary Preview")
        self.master_video = self._video_label()
        self.secondary_raw_video = self._video_label()
        self.secondary_preview_video = self._video_label()
        grid.addWidget(self.master_name, 0, 0)
        grid.addWidget(self.secondary_raw_name, 0, 1)
        grid.addWidget(self.secondary_preview_name, 0, 2)
        grid.addWidget(self.master_video, 1, 0)
        grid.addWidget(self.secondary_raw_video, 1, 1)
        grid.addWidget(self.secondary_preview_video, 1, 2)
        outer.addLayout(grid)

        self.summary_edit = QPlainTextEdit()
        self.summary_edit.setReadOnly(True)
        outer.addWidget(self.summary_edit, 1)

        self.status_label = QLabel("")
        outer.addWidget(self.status_label)

        central.setLayout(outer)
        self.setCentralWidget(central)

        self.root_browse_btn.clicked.connect(self._browse_root)
        self.inspect_btn.clicked.connect(self._inspect_root)
        self.frame_slider.valueChanged.connect(self._refresh_preview)
        self.offset_spin.valueChanged.connect(self._refresh_preview)
        self.secondary_combo.currentTextChanged.connect(self._on_secondary_changed)
        self.dry_run_btn.clicked.connect(self._dry_run)
        self.run_btn.clicked.connect(self._run_compensation)
        self.undo_btn.clicked.connect(self._undo_last)

    @staticmethod
    def _video_label() -> QLabel:
        out = QLabel("No preview.")
        out.setAlignment(Qt.AlignmentFlag.AlignCenter)
        out.setMinimumSize(320, 240)
        out.setStyleSheet("background-color: #101010; color: #f5f5f5;")
        return out

    def _browse_root(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Select session/raw folder", self.root_edit.text().strip())
        if chosen:
            self.root_edit.setText(chosen)

    def _inspect_root(self) -> None:
        root = Path(self.root_edit.text().strip())
        try:
            inspection = inspect_input_folder(root)
        except Exception as exc:
            QMessageBox.critical(self, "Inspection failed", str(exc))
            self._set_status(f"Inspection failed: {exc}")
            return
        self._inspection = inspection
        self._plan = None
        self.mode_label.setText(inspection.mode.value)
        self.master_label.setText(inspection.master_camera)
        self.secondary_combo.blockSignals(True)
        self.secondary_combo.clear()
        secondaries = [cam for cam in inspection.cameras if cam != inspection.master_camera]
        self.secondary_combo.addItems(secondaries)
        self.secondary_combo.blockSignals(False)
        if self._settings.last_secondary_camera in secondaries:
            self.secondary_combo.setCurrentText(self._settings.last_secondary_camera)
        self.frame_slider.setRange(0, max(0, inspection.camera_files[inspection.master_camera].frame_count - 1))
        self._video_provider.set_paths({cam: info.video_path for cam, info in inspection.camera_files.items()})
        self.summary_edit.setPlainText(
            "\n".join([
                f"Root: {inspection.root}",
                f"Mode: {inspection.mode.value}",
                f"Master camera: {inspection.master_camera}",
                "Length rule: compensation requires the selected master/secondary pair to end at the same frame count.",
                "Auto-trim policy: the longer side is trimmed at the end when the mismatch is 100 frames or less.",
                "Detected cameras:",
                *[
                    f"  - {cam}: frames={info.frame_count} fps={info.fps:.2f} size={info.width}x{info.height}"
                    for cam, info in inspection.camera_files.items()
                ],
                "",
                "Use Dry Run before executing compensation.",
            ])
        )
        self._refresh_preview()
        self._set_status("Inspection complete.")
        log_event(_LOG, "inspection_complete", root=root, master=inspection.master_camera)

    def _on_secondary_changed(self) -> None:
        self._plan = None
        self._refresh_preview()

    def _scaled_pixmap(self, camera: str, frame_index: int, label: QLabel) -> Optional[QPixmap]:
        image = self._video_provider.read_frame(camera, frame_index)
        if image is None:
            return None
        return QPixmap.fromImage(image).scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _refresh_preview(self) -> None:
        if self._inspection is None or self.secondary_combo.count() == 0:
            return
        master = self._inspection.master_camera
        secondary = self.secondary_combo.currentText().strip()
        if not secondary:
            return
        frame_index = int(self.frame_slider.value())
        preview_index = frame_index + int(self.offset_spin.value())
        raw_index = frame_index
        self.frame_label.setText(f"Frame: {frame_index}")
        self.raw_index_label.setText(f"Raw secondary idx: {raw_index}")
        self.preview_index_label.setText(f"Preview secondary idx: {preview_index}")
        self.master_name.setText(f"Master: {master}")
        self.secondary_raw_name.setText(f"Secondary raw: {secondary} @ {raw_index}")
        self.secondary_preview_name.setText(f"Secondary preview: {secondary} @ {preview_index}")

        pix = self._scaled_pixmap(master, frame_index, self.master_video)
        if pix is None:
            self.master_video.setText("Master frame unavailable.")
        else:
            self.master_video.setPixmap(pix)

        pix = self._scaled_pixmap(secondary, raw_index, self.secondary_raw_video)
        if pix is None:
            self.secondary_raw_video.setText("Raw secondary frame unavailable.")
        else:
            self.secondary_raw_video.setPixmap(pix)

        pix = self._scaled_pixmap(secondary, preview_index, self.secondary_preview_video)
        if pix is None:
            self.secondary_preview_video.setText("Compensated preview frame unavailable.")
        else:
            self.secondary_preview_video.setPixmap(pix)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_preview()

    def _ensure_plan(self) -> CompensationPlan:
        if self._inspection is None:
            raise RuntimeError("Inspect a folder first.")
        secondary = self.secondary_combo.currentText().strip()
        if not secondary:
            raise RuntimeError("Choose a secondary camera first.")
        plan = inspect_and_plan(self._inspection.root, secondary, int(self.offset_spin.value()))
        self._plan = plan
        return plan

    def _dry_run(self) -> None:
        try:
            plan = self._ensure_plan()
            summary = summarize_plan(plan)
        except Exception as exc:
            QMessageBox.critical(self, "Dry run failed", str(exc))
            self._set_status(f"Dry run failed: {exc}")
            return
        self.summary_edit.setPlainText(summary.details)
        self._set_status("Dry run complete.")

    def _run_compensation(self) -> None:
        try:
            plan = self._ensure_plan()
            summary = summarize_plan(plan)
        except Exception as exc:
            QMessageBox.critical(self, "Plan failed", str(exc))
            return
        if summary.rewrite_count == 0 and summary.invalidate_count == 0:
            QMessageBox.information(self, "Nothing to do", "No changes were planned for the selected camera.")
            return
        msg = (
            f"Proceed with in-place compensation?\n\n"
            f"Secondary: {plan.secondary_camera}\n"
            f"Offset: {plan.offset}\n"
            f"Rewrites: {summary.rewrite_count}\n"
            f"Invalidations: {summary.invalidate_count}\n\n"
            f"A transaction backup will be created automatically."
        )
        if QMessageBox.question(self, "Confirm compensation", msg) != QMessageBox.StandardButton.Yes:
            return
        self.summary_edit.clear()
        try:
            manifest_path = execute_plan(plan, progress=self._append_summary)
        except Exception as exc:
            QMessageBox.critical(self, "Compensation failed", str(exc))
            self._set_status(f"Compensation failed: {exc}")
            return
        self._append_summary(f"Manifest written: {manifest_path}")
        self._set_status("Compensation completed successfully.")
        self._inspect_root()

    def _undo_last(self) -> None:
        root = Path(self.root_edit.text().strip())
        if not root.is_dir():
            QMessageBox.information(self, "Undo unavailable", "Select a valid folder first.")
            return
        if QMessageBox.question(self, "Confirm undo", "Restore the most recent completed compensation backup?") != QMessageBox.StandardButton.Yes:
            return
        self.summary_edit.clear()
        try:
            manifest_path = undo_last_transaction(root, progress=self._append_summary)
        except Exception as exc:
            QMessageBox.critical(self, "Undo failed", str(exc))
            self._set_status(f"Undo failed: {exc}")
            return
        self._append_summary(f"Undo manifest updated: {manifest_path}")
        self._set_status("Undo completed successfully.")
        self._inspect_root()

    def _append_summary(self, text: str) -> None:
        self.summary_edit.appendPlainText(text)
        self.summary_edit.ensureCursorVisible()

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)
        self.statusBar().showMessage(text, 5000)

    def closeEvent(self, event) -> None:
        self._settings.last_root = self.root_edit.text().strip()
        self._settings.last_secondary_camera = self.secondary_combo.currentText().strip()
        self._settings.last_offset = int(self.offset_spin.value())
        save_settings(self._settings)
        self._video_provider.clear()
        super().closeEvent(event)
