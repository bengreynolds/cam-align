from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence, QPixmap, QShortcut
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
from cam_align_tool.core.postcheck import run_post_process_check
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
        self.offset_spin.setValue(settings.last_offset)
        self.root_edit.setText(settings.last_root)
        self._set_status("Select a session/raw folder; it will inspect immediately.")
        if settings.last_root and Path(settings.last_root).is_dir():
            self._inspect_root()

    def _build_ui(self) -> None:
        central = QWidget()
        outer = QVBoxLayout()

        self.root_edit = QLineEdit()
        self.root_browse_btn = QPushButton("Browse...")
        self.master_label = QLabel("-")
        self.mode_label = QLabel("-")
        self.secondary_combo = QComboBox()
        self.offset_spin = QSpinBox()
        self.offset_spin.setRange(-5000, 5000)
        self.offset_spin.setSingleStep(1)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_jump_spin = QSpinBox()
        self.frame_jump_spin.setRange(0, 0)
        self.frame_jump_spin.setKeyboardTracking(False)
        self.frame_jump_spin.setPrefix("#")
        self.frame_go_btn = QPushButton("Go")
        self.frame_back_chunk_btn = QPushButton("-50")
        self.frame_back_fast_btn = QPushButton("-10")
        self.frame_back_one_btn = QPushButton("-1")
        self.frame_forward_one_btn = QPushButton("+1")
        self.frame_forward_fast_btn = QPushButton("+10")
        self.frame_forward_chunk_btn = QPushButton("+50")
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(5, 5000)
        self.chunk_size_spin.setSingleStep(5)
        self.chunk_size_spin.setValue(50)
        self.epoch_combo = QComboBox()
        self.epoch_combo.setEnabled(False)
        self.epoch_combo.addItem("No pellet epochs")
        self.prev_epoch_btn = QPushButton("Prev Epoch")
        self.next_epoch_btn = QPushButton("Next Epoch")
        self.prev_epoch_btn.setEnabled(False)
        self.next_epoch_btn.setEnabled(False)
        self.frame_label = QLabel("Frame: 0")
        self.raw_index_label = QLabel("Raw secondary idx: -")
        self.preview_index_label = QLabel("Preview secondary idx: -")
        self.sign_label = QLabel("Offset convention: compensated[t] = raw[t + offset]")
        self.sign_label.setStyleSheet("color: #b71c1c; font-weight: 600;")
        self.nav_hint_label = QLabel("Hotkeys: Left/Right=1, Shift+Left/Right=10, Ctrl+Left/Right=chunk, Home/End=start/end")
        self.nav_hint_label.setStyleSheet("color: #455a64;")

        self.dry_run_btn = QPushButton("Dry Run")
        self.run_btn = QPushButton("Run Offset Compensation")
        self.post_check_btn = QPushButton("Post-Process Check")
        self.undo_btn = QPushButton("Undo Last Compensation")

        top_form = QFormLayout()
        root_row = QWidget()
        root_row_layout = QHBoxLayout()
        root_row_layout.setContentsMargins(0, 0, 0, 0)
        root_row_layout.addWidget(self.root_edit, 1)
        root_row_layout.addWidget(self.root_browse_btn)
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
        nav_row = QWidget()
        nav_row_layout = QHBoxLayout()
        nav_row_layout.setContentsMargins(0, 0, 0, 0)
        nav_row_layout.addWidget(self.frame_back_chunk_btn)
        nav_row_layout.addWidget(self.frame_back_fast_btn)
        nav_row_layout.addWidget(self.frame_back_one_btn)
        nav_row_layout.addWidget(self.frame_forward_one_btn)
        nav_row_layout.addWidget(self.frame_forward_fast_btn)
        nav_row_layout.addWidget(self.frame_forward_chunk_btn)
        nav_row_layout.addSpacing(12)
        nav_row_layout.addWidget(QLabel("Jump to:"))
        nav_row_layout.addWidget(self.frame_jump_spin)
        nav_row_layout.addWidget(self.frame_go_btn)
        nav_row_layout.addSpacing(12)
        nav_row_layout.addWidget(QLabel("Chunk:"))
        nav_row_layout.addWidget(self.chunk_size_spin)
        nav_row_layout.addStretch(1)
        nav_row.setLayout(nav_row_layout)
        top_form.addRow("Navigation:", nav_row)
        epoch_row = QWidget()
        epoch_row_layout = QHBoxLayout()
        epoch_row_layout.setContentsMargins(0, 0, 0, 0)
        epoch_row_layout.addWidget(self.prev_epoch_btn)
        epoch_row_layout.addWidget(self.epoch_combo, 1)
        epoch_row_layout.addWidget(self.next_epoch_btn)
        epoch_row.setLayout(epoch_row_layout)
        top_form.addRow("Pellet epochs:", epoch_row)
        top_form.addRow("", self.nav_hint_label)
        top_form.addRow("", self.raw_index_label)
        top_form.addRow("", self.preview_index_label)

        actions = QWidget()
        actions_layout = QHBoxLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.addWidget(self.dry_run_btn)
        actions_layout.addWidget(self.run_btn)
        actions_layout.addWidget(self.post_check_btn)
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
        self._build_menu()

        self.root_browse_btn.clicked.connect(self._browse_root)
        self.root_edit.returnPressed.connect(self._load_root_from_edit)
        self.root_edit.editingFinished.connect(self._load_root_from_edit)
        self.frame_slider.valueChanged.connect(self._refresh_preview)
        self.frame_jump_spin.valueChanged.connect(self._jump_to_frame)
        self.frame_go_btn.clicked.connect(self._jump_to_entered_frame)
        self.frame_back_one_btn.clicked.connect(lambda: self._step_frame(-1))
        self.frame_forward_one_btn.clicked.connect(lambda: self._step_frame(1))
        self.frame_back_fast_btn.clicked.connect(lambda: self._step_frame(-10))
        self.frame_forward_fast_btn.clicked.connect(lambda: self._step_frame(10))
        self.frame_back_chunk_btn.clicked.connect(lambda: self._step_frame(-self._chunk_size()))
        self.frame_forward_chunk_btn.clicked.connect(lambda: self._step_frame(self._chunk_size()))
        self.chunk_size_spin.valueChanged.connect(self._update_chunk_button_labels)
        self.epoch_combo.currentIndexChanged.connect(self._jump_to_selected_epoch)
        self.prev_epoch_btn.clicked.connect(self._jump_to_prev_epoch)
        self.next_epoch_btn.clicked.connect(self._jump_to_next_epoch)
        self.offset_spin.valueChanged.connect(self._refresh_preview)
        self.secondary_combo.currentTextChanged.connect(self._on_secondary_changed)
        self.dry_run_btn.clicked.connect(self._dry_run)
        self.run_btn.clicked.connect(self._run_compensation)
        self.post_check_btn.clicked.connect(self._post_process_check)
        self.undo_btn.clicked.connect(self._undo_last)
        self._build_shortcuts()
        self._update_chunk_button_labels()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        browse_action = QAction("Browse Session...", self)
        browse_action.triggered.connect(self._browse_root)
        file_menu.addAction(browse_action)

        reload_action = QAction("Reload Session", self)
        reload_action.triggered.connect(self._reload_current_root)
        file_menu.addAction(reload_action)

        file_menu.addSeparator()

        self.backup_action = QAction("Create Transaction Backup", self)
        self.backup_action.setCheckable(True)
        self.backup_action.setChecked(bool(self._settings.create_backup))
        self.backup_action.toggled.connect(self._on_backup_toggled)
        file_menu.addAction(self.backup_action)

        self.auto_post_check_action = QAction("Auto-Run Post-Process Check", self)
        self.auto_post_check_action.setCheckable(True)
        self.auto_post_check_action.setChecked(bool(self._settings.auto_post_check))
        self.auto_post_check_action.toggled.connect(self._on_auto_post_check_toggled)
        file_menu.addAction(self.auto_post_check_action)

        file_menu.addSeparator()

        default_root_action = QAction("Set Default Browse Root...", self)
        default_root_action.triggered.connect(self._select_default_browse_root)
        file_menu.addAction(default_root_action)

        ffprobe_action = QAction("Set ffprobe Path...", self)
        ffprobe_action.triggered.connect(self._select_ffprobe_path)
        file_menu.addAction(ffprobe_action)

        clear_ffprobe_action = QAction("Clear ffprobe Path", self)
        clear_ffprobe_action.triggered.connect(self._clear_ffprobe_path)
        file_menu.addAction(clear_ffprobe_action)

        file_menu.addSeparator()

        hotkeys_action = QAction("Hotkey Reference", self)
        hotkeys_action.triggered.connect(self._show_hotkey_reference)
        file_menu.addAction(hotkeys_action)

    def _build_shortcuts(self) -> None:
        bindings = [
            ("Left", lambda: self._step_frame(-1)),
            ("Right", lambda: self._step_frame(1)),
            ("Shift+Left", lambda: self._step_frame(-10)),
            ("Shift+Right", lambda: self._step_frame(10)),
            ("Ctrl+Left", lambda: self._step_frame(-self._chunk_size())),
            ("Ctrl+Right", lambda: self._step_frame(self._chunk_size())),
            ("Home", self._jump_to_start),
            ("End", self._jump_to_end),
        ]
        self._shortcuts: list[QShortcut] = []
        for sequence, callback in bindings:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

    def _show_hotkey_reference(self) -> None:
        text = (
            "Navigation hotkeys:\n\n"
            "Left / Right: step by 1 frame\n"
            "Shift+Left / Shift+Right: step by 10 frames\n"
            "Ctrl+Left / Ctrl+Right: step by the current chunk size\n"
            "Home / End: jump to first / last frame\n\n"
            "Controls:\n\n"
            "Jump to: enter an exact master-frame index and press Go\n"
            "Chunk: sets the size used by chunk jump buttons and Ctrl+Left / Ctrl+Right"
        )
        QMessageBox.information(self, "Hotkey Reference", text)

    @staticmethod
    def _video_label() -> QLabel:
        out = QLabel("No preview.")
        out.setAlignment(Qt.AlignmentFlag.AlignCenter)
        out.setMinimumSize(320, 240)
        out.setStyleSheet("background-color: #101010; color: #f5f5f5;")
        return out

    def _save_settings(self) -> None:
        save_settings(self._settings)

    def _max_frame_index(self) -> int:
        return int(self.frame_slider.maximum())

    def _chunk_size(self) -> int:
        return int(self.chunk_size_spin.value())

    def _update_chunk_button_labels(self) -> None:
        chunk = self._chunk_size()
        self.frame_back_chunk_btn.setText(f"-{chunk}")
        self.frame_forward_chunk_btn.setText(f"+{chunk}")
        self.frame_slider.setSingleStep(1)
        self.frame_slider.setPageStep(chunk)

    def _set_frame_index(self, frame_index: int) -> None:
        bounded = max(0, min(int(frame_index), self._max_frame_index()))
        if bounded == int(self.frame_slider.value()):
            self._refresh_preview()
            return
        self.frame_slider.setValue(bounded)

    def _step_frame(self, delta: int) -> None:
        if self._inspection is None:
            return
        self._set_frame_index(int(self.frame_slider.value()) + int(delta))

    def _jump_to_frame(self, frame_index: int) -> None:
        if self._inspection is None:
            return
        self._set_frame_index(frame_index)

    def _jump_to_entered_frame(self) -> None:
        self._jump_to_frame(int(self.frame_jump_spin.value()))

    def _jump_to_start(self) -> None:
        self._set_frame_index(0)

    def _jump_to_end(self) -> None:
        self._set_frame_index(self._max_frame_index())

    def _epoch_count(self) -> int:
        if self._inspection is None:
            return 0
        return len(self._inspection.reach_epochs)

    def _populate_epoch_selector(self) -> None:
        self.epoch_combo.blockSignals(True)
        self.epoch_combo.clear()
        epochs = self._inspection.reach_epochs if self._inspection is not None else ()
        if epochs:
            self.epoch_combo.addItem("Select pellet epoch", None)
            for epoch in epochs:
                self.epoch_combo.addItem(epoch.label, epoch.pellet_frame)
            self.epoch_combo.setEnabled(True)
            self.prev_epoch_btn.setEnabled(True)
            self.next_epoch_btn.setEnabled(True)
            self.epoch_combo.setCurrentIndex(0)
        else:
            self.epoch_combo.addItem("No pellet epochs")
            self.epoch_combo.setEnabled(False)
            self.prev_epoch_btn.setEnabled(False)
            self.next_epoch_btn.setEnabled(False)
        self.epoch_combo.blockSignals(False)

    def _jump_to_selected_epoch(self, index: int) -> None:
        if self._inspection is None:
            return
        if index <= 0 or index > len(self._inspection.reach_epochs):
            return
        epoch = self._inspection.reach_epochs[index - 1]
        self._log_ui(f"Jumping to {epoch.label}.")
        self._set_frame_index(epoch.pellet_frame)

    def _jump_to_prev_epoch(self) -> None:
        count = self._epoch_count()
        if count == 0:
            return
        current = self.epoch_combo.currentIndex()
        if current <= 1:
            self.epoch_combo.setCurrentIndex(1)
            return
        self.epoch_combo.setCurrentIndex(current - 1)

    def _jump_to_next_epoch(self) -> None:
        count = self._epoch_count()
        if count == 0:
            return
        current = self.epoch_combo.currentIndex()
        if current <= 0:
            self.epoch_combo.setCurrentIndex(1)
            return
        self.epoch_combo.setCurrentIndex(min(count, current + 1))

    def _browse_start_dir(self) -> str:
        current = Path(self.root_edit.text().strip())
        if current.is_dir():
            return str(current)
        configured = Path(self._settings.default_browse_root.strip())
        if self._settings.default_browse_root and configured.is_dir():
            return str(configured)
        return str(Path.cwd())

    def _browse_root(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Select session/raw folder", self._browse_start_dir())
        if chosen:
            self._set_root_path(chosen)

    def _set_root_path(self, root_text: str, reset_log: bool = True) -> None:
        root = Path(root_text).expanduser()
        self.root_edit.setText(str(root))
        self._settings.last_root = str(root)
        self._save_settings()
        if root.is_dir():
            self._inspect_root(reset_log=reset_log)
        else:
            self._set_status(f"Selected folder does not exist: {root}")

    def _load_root_from_edit(self) -> None:
        root_text = self.root_edit.text().strip()
        if not root_text:
            return
        root = Path(root_text).expanduser()
        if self._inspection is not None and root == self._inspection.root:
            return
        self._set_root_path(root_text)

    def _reload_current_root(self) -> None:
        root_text = self.root_edit.text().strip()
        if not root_text:
            QMessageBox.information(self, "Reload unavailable", "Select a valid folder first.")
            return
        root = Path(root_text).expanduser()
        if not root.is_dir():
            QMessageBox.information(self, "Reload unavailable", "Select a valid folder first.")
            return
        self._inspect_root()

    def _select_default_browse_root(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Select default session browse root", self._browse_start_dir())
        if not chosen:
            return
        self._settings.default_browse_root = chosen
        self._save_settings()
        self._log_ui(f"Default browse root set to {chosen}")

    def _select_ffprobe_path(self) -> None:
        initial = self._settings.ffprobe_path.strip() or self._browse_start_dir()
        chosen, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select ffprobe executable",
            initial,
            "Executables (*.exe *.bat *.cmd);;All files (*)",
        )
        if not chosen:
            return
        self._settings.ffprobe_path = chosen
        self._save_settings()
        self._log_ui(f"ffprobe path set to {chosen}")

    def _clear_ffprobe_path(self) -> None:
        self._settings.ffprobe_path = ""
        self._save_settings()
        self._log_ui("ffprobe path cleared; automatic search will be used.")

    def _on_backup_toggled(self, checked: bool) -> None:
        self._settings.create_backup = bool(checked)
        self._save_settings()
        self._log_ui(f"Create Transaction Backup set to {self._settings.create_backup}.")

    def _on_auto_post_check_toggled(self, checked: bool) -> None:
        self._settings.auto_post_check = bool(checked)
        self._save_settings()
        self._log_ui(f"Auto-Run Post-Process Check set to {self._settings.auto_post_check}.")

    @staticmethod
    def _log_stamp() -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _log_ui(self, text: str) -> None:
        for line in str(text).splitlines() or [""]:
            self.summary_edit.appendPlainText(f"[{self._log_stamp()}] {line}")
        self.summary_edit.ensureCursorVisible()

    def _start_log(self, title: str) -> None:
        self.summary_edit.clear()
        self._log_ui(title)

    def _log_report(self, title: str, text: str) -> None:
        self._log_ui(title)
        for line in text.splitlines():
            self._log_ui(f"    {line}")

    def _inspect_root(self, reset_log: bool = True) -> None:
        root = Path(self.root_edit.text().strip())
        if reset_log:
            self._start_log(f"Inspecting {root}")
        else:
            self._log_ui(f"Refreshing inspection for {root}")
        try:
            self._log_ui("Reading systemdata, camera videos, timestamps, and dropped-frame warnings.")
            inspection = inspect_input_folder(root)
        except Exception as exc:
            self._log_ui(f"Inspection failed: {exc}")
            QMessageBox.critical(self, "Inspection failed", str(exc))
            self._set_status(f"Inspection failed: {exc}")
            return
        self._inspection = inspection
        self._plan = None
        self._settings.last_root = str(inspection.root)
        self._save_settings()
        self._log_ui(f"Inspection succeeded. Master={inspection.master_camera}; cameras={', '.join(inspection.cameras)}")
        self.mode_label.setText(inspection.mode.value)
        self.master_label.setText(inspection.master_camera)
        self.secondary_combo.blockSignals(True)
        self.secondary_combo.clear()
        secondaries = [cam for cam in inspection.cameras if cam != inspection.master_camera]
        self.secondary_combo.addItems(secondaries)
        self.secondary_combo.blockSignals(False)
        if self._settings.last_secondary_camera in secondaries:
            self.secondary_combo.setCurrentText(self._settings.last_secondary_camera)
        self._populate_epoch_selector()
        max_frame = max(0, inspection.camera_files[inspection.master_camera].frame_count - 1)
        self.frame_slider.setRange(0, max_frame)
        self.frame_jump_spin.blockSignals(True)
        self.frame_jump_spin.setRange(0, max_frame)
        self.frame_jump_spin.setValue(min(int(self.frame_slider.value()), max_frame))
        self.frame_jump_spin.blockSignals(False)
        self._video_provider.set_paths({cam: info.video_path for cam, info in inspection.camera_files.items()})
        self._log_report(
            "Inspection summary",
            "\n".join([
                f"Root: {inspection.root}",
                f"Mode: {inspection.mode.value}",
                f"Master camera: {inspection.master_camera}",
                "Alignment rule: the systemdata-selected master stays authoritative; the other non-cam3 camera is shifted first.",
                "Length rule: after the shift, the secondary is trimmed or buffered at the tail to match the master when the mismatch is 100 frames or less.",
                "Error rule: if the master/secondary mismatch exceeds 100 frames, the run stops as a significant acquisition alignment error.",
                "Timestamp rule: the secondary timestamps file is rewritten with the same shift/length policy when present.",
                (
                    f"Pellet epoch navigation: {len(inspection.reach_epochs)} pellet-delivery epoch(s) available from events.txt."
                    if inspection.reach_epochs
                    else "Pellet epoch navigation: unavailable because no pellet_delivery entries were found in events.txt."
                ),
                (
                    "Reach annotations: enabled from reaches.txt when present."
                    if inspection.reaches_file_present
                    else "Reach annotations: unavailable because reaches.txt was not found."
                ),
                "Detected cameras:",
                *[
                    (
                        f"  - {cam}: frames={info.frame_count} fps={info.fps:.2f} size={info.width}x{info.height}"
                        + (f" dropped_frames={len(info.dropped_frames)}" if len(info.dropped_frames) > 0 else "")
                    )
                    for cam, info in inspection.camera_files.items()
                ],
                *(["", "Warnings:", *[f"  - {warning}" for warning in inspection.warnings]] if inspection.warnings else []),
                "",
                "Use Dry Run before executing compensation.",
            ]),
        )
        self._refresh_preview()
        self._set_status("Inspection complete.")
        if inspection.warnings:
            self._log_ui("Inspection warnings were found; user confirmation dialog will be shown.")
            QMessageBox.warning(
                self,
                "Dropped-frame warning",
                "Timestamp inspection found possible in-recording dropped frames or timestamp inconsistencies.\n\n"
                "This tool only fixes startup/end offset mismatch. Review the warning list in the summary before running compensation.",
            )
        log_event(_LOG, "inspection_complete", root=root, master=inspection.master_camera)

    def _on_secondary_changed(self) -> None:
        self._plan = None
        if self._inspection is not None:
            secondary = self.secondary_combo.currentText().strip()
            if secondary:
                self._log_ui(f"Secondary camera changed to {secondary}.")
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
        self.frame_jump_spin.blockSignals(True)
        self.frame_jump_spin.setValue(frame_index)
        self.frame_jump_spin.blockSignals(False)
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
        self._start_log("Starting dry run")
        try:
            self._log_ui("Building compensation plan.")
            plan = self._ensure_plan()
            summary = summarize_plan(plan)
        except Exception as exc:
            self._log_ui(f"Dry run failed: {exc}")
            QMessageBox.critical(self, "Dry run failed", str(exc))
            self._set_status(f"Dry run failed: {exc}")
            return
        self._log_report("Dry-run summary", summary.details)
        self._set_status("Dry run complete.")

    def _run_compensation(self) -> None:
        self._start_log("Starting compensation run")
        try:
            self._log_ui("Building final compensation plan.")
            plan = self._ensure_plan()
            summary = summarize_plan(plan)
        except Exception as exc:
            self._log_ui(f"Plan failed: {exc}")
            QMessageBox.critical(self, "Plan failed", str(exc))
            return
        if summary.rewrite_count == 0 and summary.invalidate_count == 0:
            self._log_ui("Nothing to do; no rewrites or invalidations were planned.")
            QMessageBox.information(self, "Nothing to do", "No changes were planned for the selected camera.")
            return
        self._log_report("Planned run summary", summary.details)
        msg = (
            f"Proceed with in-place compensation?\n\n"
            f"Secondary: {plan.secondary_camera}\n"
            f"Offset: {plan.offset}\n"
            f"Rewrites: {summary.rewrite_count}\n"
            f"Invalidations: {summary.invalidate_count}\n\n"
        )
        if self._settings.create_backup:
            msg += "A transaction backup will be created automatically."
        else:
            msg += "Backup creation is disabled. Files will be changed in place and undo may not be available."
            if summary.invalidate_count > 0:
                msg += "\n\nWarning: this run will invalidate derived files without creating a restorable backup."
        if plan.inspection.warnings:
            self._log_ui("Plan contains inspection warnings; run confirmation will include them.")
            msg += (
                "\n\nWarning: timestamp inspection found possible in-recording dropped frames or timestamp inconsistencies. "
                "This tool does not repair those conditions."
            )
        if QMessageBox.question(self, "Confirm compensation", msg) != QMessageBox.StandardButton.Yes:
            self._log_ui("Compensation cancelled by user.")
            return
        self._start_log("Executing compensation")
        try:
            manifest_path = execute_plan(plan, progress=self._append_summary, create_backup=self._settings.create_backup)
        except Exception as exc:
            self._log_ui(f"Compensation failed: {exc}")
            QMessageBox.critical(self, "Compensation failed", str(exc))
            self._set_status(f"Compensation failed: {exc}")
            return
        self._append_summary(f"Manifest written: {manifest_path}")
        self._set_status("Compensation completed successfully.")
        self._log_ui("Refreshing inspection after compensation.")
        self._inspect_root(reset_log=False)
        if self._settings.auto_post_check and self._inspection is not None:
            self._log_ui("Auto post-process check is enabled; running verification now.")
            self._post_process_check(reset_log=False)

    def _post_process_check(self, reset_log: bool = True) -> None:
        if self._inspection is None:
            QMessageBox.information(self, "Post-process check unavailable", "Inspect a folder first.")
            return
        secondary = self.secondary_combo.currentText().strip()
        if not secondary:
            QMessageBox.information(self, "Post-process check unavailable", "Choose a secondary camera first.")
            return
        if reset_log:
            self._start_log(f"Starting post-process check for {secondary}")
        else:
            self._log_ui(f"Starting post-process check for {secondary}")
        try:
            report = run_post_process_check(
                self._inspection,
                secondary,
                progress=self._append_summary,
                ffprobe_path=self._settings.ffprobe_path,
            )
        except Exception as exc:
            self._log_ui(f"Post-process check failed: {exc}")
            QMessageBox.critical(self, "Post-process check failed", str(exc))
            self._set_status(f"Post-process check failed: {exc}")
            return
        self._log_report("Post-process report", report.details)
        if report.passed:
            self._log_ui("Post-process check passed.")
            self._set_status("Post-process check passed.")
            QMessageBox.information(self, "Post-process check", "All checked dense pairs matched.")
        else:
            self._log_ui("Post-process check found mismatches.")
            self._set_status("Post-process check found mismatches.")
            QMessageBox.warning(self, "Post-process check", "One or more checked dense pairs did not match. Review the report.")

    def _undo_last(self) -> None:
        root = Path(self.root_edit.text().strip())
        if not root.is_dir():
            QMessageBox.information(self, "Undo unavailable", "Select a valid folder first.")
            return
        self._start_log(f"Starting undo for {root}")
        if QMessageBox.question(self, "Confirm undo", "Restore the most recent completed compensation backup?") != QMessageBox.StandardButton.Yes:
            self._log_ui("Undo cancelled by user.")
            return
        try:
            manifest_path = undo_last_transaction(root, progress=self._append_summary)
        except Exception as exc:
            self._log_ui(f"Undo failed: {exc}")
            QMessageBox.critical(self, "Undo failed", str(exc))
            self._set_status(f"Undo failed: {exc}")
            return
        self._append_summary(f"Undo manifest updated: {manifest_path}")
        self._set_status("Undo completed successfully.")
        self._log_ui("Refreshing inspection after undo.")
        self._inspect_root(reset_log=False)

    def _append_summary(self, text: str) -> None:
        self._log_ui(text)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)
        self.statusBar().showMessage(text, 5000)

    def closeEvent(self, event) -> None:
        self._settings.last_root = self.root_edit.text().strip()
        self._settings.last_secondary_camera = self.secondary_combo.currentText().strip()
        self._settings.last_offset = int(self.offset_spin.value())
        self._save_settings()
        self._video_provider.clear()
        super().closeEvent(event)
