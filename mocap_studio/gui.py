"""PySide6 GUI for Mocap Studio."""

from __future__ import annotations

import sys

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox,
                               QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QPushButton, QSlider, QSpinBox,
                               QVBoxLayout, QWidget)

from .camera import enumerate_cameras
from .config import Settings
from .tracker import TrackerWorker
from .vrm_loader import load_vrm


class SmoothingSlider(QWidget):
    """Labeled 0-100% slider bound to a Settings float field."""

    def __init__(self, label: str, value: float, on_change) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel(label)
        self._label.setMinimumWidth(90)
        self._value_label = QLabel(f"{int(value * 100)}%")
        self._value_label.setMinimumWidth(40)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 100)
        self._slider.setValue(int(value * 100))
        self._on_change = on_change

        def _changed(v: int) -> None:
            self._value_label.setText(f"{v}%")
            self._on_change(v / 100.0)

        self._slider.valueChanged.connect(_changed)
        lay.addWidget(self._label)
        lay.addWidget(self._slider, 1)
        lay.addWidget(self._value_label)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Mocap Studio — NVIDIA AR + MediaPipe トラッカー")
        self.settings = Settings.load()
        self.worker = TrackerWorker(self.settings)
        self._error_shown = False

        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        # ---------------- left: preview + status ----------------
        left = QVBoxLayout()
        self.preview = QLabel("プレビュー")
        self.preview.setMinimumSize(480, 360)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet(
            "background:#202020;color:#888;border:1px solid #444;")
        left.addWidget(self.preview, 1)

        self.status_label = QLabel("停止中")
        left.addWidget(self.status_label)

        row = QHBoxLayout()
        self.start_btn = QPushButton("▶ トラッキング開始")
        self.start_btn.setMinimumHeight(36)
        self.start_btn.clicked.connect(self._toggle)
        row.addWidget(self.start_btn)
        self.calib_btn = QPushButton("顔キャリブレーション（真顔で押す）")
        self.calib_btn.clicked.connect(self.worker.request_face_calibration)
        row.addWidget(self.calib_btn)
        self.body_calib_btn = QPushButton(
            "姿勢・指キャリブレーション（楽な姿勢・手をパーにして押す）")
        self.body_calib_btn.clicked.connect(
            self.worker.request_body_calibration)
        row.addWidget(self.body_calib_btn)
        left.addLayout(row)
        checks = QHBoxLayout()
        self.mirror_check = QCheckBox("ミラーモード（鏡のように動く）")
        self.mirror_check.setChecked(self.settings.mirror_tracking)
        self.mirror_check.toggled.connect(self._set_mirror)
        checks.addWidget(self.mirror_check)
        self.preview_check = QCheckBox("プレビュー表示（軽くしたい時はOFF）")
        self.preview_check.setChecked(True)
        self.preview_check.toggled.connect(self.worker.set_preview_enabled)
        checks.addWidget(self.preview_check)
        left.addLayout(checks)
        layout.addLayout(left, 2)

        # ---------------- right: settings ----------------
        right = QVBoxLayout()

        cam_group = QGroupBox("カメラ")
        cam_form = QFormLayout(cam_group)
        self.cam_combo = QComboBox()
        self._populate_cameras()
        cam_form.addRow("デバイス", self.cam_combo)
        self.res_combo = QComboBox()
        for label in ("1280x720", "1920x1080", "960x540", "640x480"):
            self.res_combo.addItem(label)
        self.res_combo.setCurrentText(
            f"{self.settings.camera_width}x{self.settings.camera_height}")
        cam_form.addRow("解像度", self.res_combo)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(15, 120)
        self.fps_spin.setValue(self.settings.camera_fps)
        cam_form.addRow("FPS", self.fps_spin)
        right.addWidget(cam_group)

        face_group = QGroupBox("フェイシャル（iFacialMocap v1 / パーフェクトシンク）")
        face_group.setCheckable(True)
        face_group.setChecked(self.settings.face_enabled)
        face_form = QFormLayout(face_group)
        self.face_host = QLineEdit(self.settings.face_host)
        face_form.addRow("送信先IP", self.face_host)
        self.face_port = QSpinBox()
        self.face_port.setRange(1, 65535)
        self.face_port.setValue(self.settings.face_port)
        face_form.addRow("ポート", self.face_port)
        face_form.addRow(SmoothingSlider(
            "表情の平滑化", self.settings.face_smoothing,
            lambda v: self._set_smoothing("face_smoothing", v)))
        face_form.addRow(SmoothingSlider(
            "頭の平滑化", self.settings.head_smoothing,
            lambda v: self._set_smoothing("head_smoothing", v)))
        self.face_group = face_group
        right.addWidget(face_group)

        body_group = QGroupBox("上半身（VMCプロトコル）")
        body_group.setCheckable(True)
        body_group.setChecked(self.settings.body_enabled)
        body_form = QFormLayout(body_group)
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("MediaPipe（体＋指の完全トラッキング）",
                                   "mediapipe")
        self.backend_combo.addItem("NVIDIA（GPU・指は開閉推定のみ）", "nvidia")
        body_form.addRow("バックエンド", self.backend_combo)
        self.vmc_host = QLineEdit(self.settings.vmc_host)
        body_form.addRow("送信先IP", self.vmc_host)
        self.vmc_port = QSpinBox()
        self.vmc_port.setRange(1, 65535)
        self.vmc_port.setValue(self.settings.vmc_port)
        body_form.addRow("ポート", self.vmc_port)
        body_form.addRow(SmoothingSlider(
            "体の平滑化", self.settings.body_smoothing,
            lambda v: self._set_smoothing("body_smoothing", v)))
        body_form.addRow(SmoothingSlider(
            "指の平滑化", self.settings.finger_smoothing,
            lambda v: self._set_smoothing("finger_smoothing", v)))
        vrm_row = QHBoxLayout()
        self.vrm_label = QLabel("（未読込：内蔵デフォルト使用）")
        self.vrm_label.setWordWrap(True)
        vrm_btn = QPushButton("VRM読込")
        vrm_btn.clicked.connect(self._load_vrm_dialog)
        vrm_row.addWidget(self.vrm_label, 1)
        vrm_row.addWidget(vrm_btn)
        body_form.addRow("モデル", vrm_row)
        self.body_group = body_group
        right.addWidget(body_group)

        right.addStretch(1)
        note = QLabel("※ 顔・頭は iFacialMocap 側、体・指は VMC 側で送信。\n"
                      "受信アプリ側で両プロトコルの受信を有効にしてください。")
        note.setStyleSheet("color:#888;")
        right.addWidget(note)
        layout.addLayout(right, 1)

        if self.settings.vrm_model_path:
            self._load_vrm(self.settings.vrm_model_path)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(33)

    # ------------------------------------------------------------------
    def _populate_cameras(self) -> None:
        cams = enumerate_cameras()
        if not cams:
            cams = [0]
        for i in cams:
            self.cam_combo.addItem(f"カメラ {i}", i)
        idx = self.cam_combo.findData(self.settings.camera_index)
        if idx >= 0:
            self.cam_combo.setCurrentIndex(idx)

    def _set_smoothing(self, field: str, value: float) -> None:
        setattr(self.settings, field, value)
        self.worker.apply_smoothing()

    def _set_mirror(self, on: bool) -> None:
        self.settings.mirror_tracking = on

    def _load_vrm_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "VRMモデルを選択", "", "VRM (*.vrm)")
        if path:
            self._load_vrm(path)

    def _load_vrm(self, path: str) -> None:
        try:
            info = load_vrm(path)
        except Exception as e:
            QMessageBox.warning(self, "VRM読込エラー", str(e))
            return
        self.worker.body_retargeter.set_bone_offsets(info.bone_offsets)
        self.settings.vrm_model_path = path
        ps = "✅ パーフェクトシンク対応" if info.perfect_sync \
            else "⚠ ARKitモーフ無し"
        self.vrm_label.setText(
            f"{info.model_name} (VRM {info.spec_version}) {ps}")

    def _apply_settings_from_ui(self) -> None:
        s = self.settings
        s.camera_index = self.cam_combo.currentData() or 0
        w, h = self.res_combo.currentText().split("x")
        s.camera_width, s.camera_height = int(w), int(h)
        s.camera_fps = self.fps_spin.value()
        s.face_enabled = self.face_group.isChecked()
        s.face_host = self.face_host.text().strip() or "127.0.0.1"
        s.face_port = self.face_port.value()
        s.body_enabled = self.body_group.isChecked()
        s.vmc_host = self.vmc_host.text().strip() or "127.0.0.1"
        s.vmc_port = self.vmc_port.value()
        self.worker.body_backend = self.backend_combo.currentData()

    # ------------------------------------------------------------------
    def _toggle(self) -> None:
        st = self.worker.get_status()
        if st.running:
            self.start_btn.setEnabled(False)
            self.start_btn.setText("停止中...")
            self.worker.stop()
            self.start_btn.setEnabled(True)
            self.start_btn.setText("▶ トラッキング開始")
        else:
            self._apply_settings_from_ui()
            if not (self.settings.face_enabled or self.settings.body_enabled):
                QMessageBox.information(
                    self, "確認",
                    "フェイシャル・上半身のどちらかを有効にしてください。")
                return
            self._error_shown = False
            self.worker.start()
            self.start_btn.setText("■ 停止")

    def _poll(self) -> None:
        st = self.worker.get_status()
        if st.running:
            face = "😀" if st.face_found else "–"
            body = "🧍" if st.body_found else "–"
            lh = "🫲" if st.hands_found[0] else "–"
            rh = "🫱" if st.hands_found[1] else "–"
            self.status_label.setText(
                f"{st.info}  |  {st.fps:.0f} FPS  |  顔:{face} 体:{body} "
                f"左手:{lh} 右手:{rh}")
        else:
            self.status_label.setText(st.info or "停止中")
            if self.start_btn.text() != "▶ トラッキング開始":
                self.start_btn.setText("▶ トラッキング開始")
        if st.error and not self._error_shown:
            self._error_shown = True
            QMessageBox.critical(self, "トラッキングエラー", st.error)
        if st.preview is not None and self.preview_check.isChecked():
            frame = st.preview
            if self.settings.mirror_preview:
                frame = cv2.flip(frame, 1)
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            h, w = rgb.shape[:2]
            img = QImage(rgb.data, w, h, 3 * w,
                         QImage.Format.Format_RGB888).copy()
            pix = QPixmap.fromImage(img).scaled(
                self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self.preview.setPixmap(pix)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._apply_settings_from_ui()
        self.settings.save()
        self.worker.stop()
        event.accept()


def run() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1100, 640)
    win.show()
    return app.exec()
