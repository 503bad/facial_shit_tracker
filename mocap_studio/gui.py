"""PySide6 GUI for Mocap Studio."""

from __future__ import annotations

import sys

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtCore import QUrl
from PySide6.QtGui import QCloseEvent, QDesktopServices, QImage, QPixmap
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
        self.camera_check = QCheckBox("カメラ映像を表示")
        self.camera_check.setChecked(self.settings.show_camera)
        self.camera_check.toggled.connect(self._set_show_camera)
        checks.addWidget(self.camera_check)
        checks.addWidget(QLabel("補間出力:"))
        self.interp_combo = QComboBox()
        self.interp_combo.addItem("OFF（直接送信）", 0)
        self.interp_combo.addItem("30FPS（僅かな遅延）", 30)
        self.interp_combo.addItem("60FPS（僅かな遅延）", 60)
        cur = self.settings.output_fps if self.settings.output_interp else 0
        idx = self.interp_combo.findData(cur)
        self.interp_combo.setCurrentIndex(max(0, idx))
        self.interp_combo.setToolTip("変更はトラッキング再開時に反映されます")
        checks.addWidget(self.interp_combo)
        checks.addWidget(QLabel("先読み補正:"))
        self.refine_combo = QComboBox()
        self.refine_combo.addItem("OFF", 0.0)
        for sec in (0.10, 0.15, 0.20, 0.30):
            self.refine_combo.addItem(f"{sec:.2f}秒遅延", sec)
        cur = (round(self.settings.output_lookahead_sec, 2)
               if self.settings.output_refine else 0.0)
        idx = self.refine_combo.findData(cur)
        self.refine_combo.setCurrentIndex(max(0, idx))
        self.refine_combo.setToolTip(
            "補間出力がONの時のみ有効。指定秒だけ遅らせ、前後のサンプルから"
            "外れ値を除いて平滑化します。変更はトラッキング再開時に反映")
        checks.addWidget(self.refine_combo)
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
        self.face_output_combo = QComboBox()
        self.face_output_combo.addItem("iFacialMocap形式（Warudo/VSeeFace等）", "ifm")
        self.face_output_combo.addItem("VMC Blend/Val（VRM4U/UE5等）", "vmc")
        self.face_output_combo.addItem("両方に送信", "both")
        idx = self.face_output_combo.findData(self.settings.face_output)
        self.face_output_combo.setCurrentIndex(max(0, idx))
        face_form.addRow("表情の送信先", self.face_output_combo)
        self.eye_mode_combo = QComboBox()
        self.eye_mode_combo.addItem("ボーン（LeftEye/RightEye）", "bone")
        self.eye_mode_combo.addItem("モーフ（eyeLook*）", "morph")
        self.eye_mode_combo.addItem("両方", "both")
        idx = self.eye_mode_combo.findData(self.settings.eye_mode)
        self.eye_mode_combo.setCurrentIndex(max(0, idx))
        face_form.addRow("眼球（VMC送信時）", self.eye_mode_combo)
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
        self.gate_check = QCheckBox("震え抑制ゲート（静止時の微振動をカット）")
        self.gate_check.setChecked(self.settings.body_gate_enabled)
        self.gate_check.toggled.connect(self._set_gate)
        body_form.addRow(self.gate_check)
        self.legs_check = QCheckBox("下半身も送信（脚が映っている時のみ動作）")
        self.legs_check.setChecked(self.settings.send_legs)
        self.legs_check.toggled.connect(self._set_send_legs)
        body_form.addRow(self.legs_check)
        self.ground_check = QCheckBox("接地IK（脚ON時：浮き・滑りを防ぎ足裏を床に固定）")
        self.ground_check.setChecked(self.settings.ground_mode)
        self.ground_check.toggled.connect(self._set_ground)
        body_form.addRow(self.ground_check)
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

        # Optional 2-camera depth extension: fully isolated; if the
        # stereo package cannot load, the app runs exactly as before.
        self.stereo_panel = None
        if self.worker.stereo_settings is not None:
            try:
                from .stereo.ui import StereoPanel
                self.stereo_panel = StereoPanel(self.worker, self)
                right.addWidget(self.stereo_panel)
            except Exception as e:
                warn = QLabel(f"2カメラ拡張を読み込めませんでした: {e}")
                warn.setWordWrap(True)
                warn.setStyleSheet("color:#c66;")
                right.addWidget(warn)

        right.addStretch(1)
        note = QLabel("※ 表情は iFacialMocap 側、体・指・首・頭は VMC 側で送信\n"
                      "（頭回転は iFacialMocap にも同時送信）。\n"
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

    def _set_show_camera(self, on: bool) -> None:
        self.settings.show_camera = on
        self.worker.set_show_camera(on)

    def _set_gate(self, on: bool) -> None:
        self.settings.body_gate_enabled = on
        self.worker.set_gate_enabled(on)

    def _set_ground(self, on: bool) -> None:
        self.settings.ground_mode = on
        self.worker.set_ground_mode(on)

    def _set_send_legs(self, on: bool) -> None:
        self.settings.send_legs = on
        self.worker.set_send_legs(on)

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
        s.face_output = self.face_output_combo.currentData() or "ifm"
        fps = int(self.interp_combo.currentData() or 0)
        s.output_interp = fps > 0
        la = float(self.refine_combo.currentData() or 0.0)
        s.output_refine = la > 0
        if la > 0:
            s.output_lookahead_sec = la
        if fps > 0:
            s.output_fps = fps
        s.eye_mode = self.eye_mode_combo.currentData() or "bone"
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
        if self.stereo_panel is not None:
            self.stereo_panel.update_status(st)
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
            self._show_error(st.error)
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

    def _show_error(self, text: str) -> None:
        """Show a readable message; for known setup problems hide the
        traceback and offer to open the download page."""
        if "NvArError" in text and "NVIDIA AR SDK" in text:
            msg = text.split("NvArError: ", 1)[-1].strip()
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Critical)
            box.setWindowTitle("NVIDIA AR SDK が必要です")
            box.setText(msg)
            open_btn = box.addButton("ダウンロードページを開く",
                                     QMessageBox.ButtonRole.ActionRole)
            box.addButton(QMessageBox.StandardButton.Close)
            box.exec()
            if box.clickedButton() is open_btn:
                from .nvar import SDK_URL
                QDesktopServices.openUrl(QUrl(SDK_URL))
            return
        QMessageBox.critical(self, "トラッキングエラー", text)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._apply_settings_from_ui()
        self.settings.save()
        if self.stereo_panel is not None:
            self.stereo_panel.save_settings()
        self.worker.stop()
        event.accept()


def run() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1100, 640)
    win.show()
    return app.exec()
