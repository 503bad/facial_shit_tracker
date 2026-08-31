"""PySide6 UI for the 2-camera depth extension.

``StereoPanel`` is a self-contained settings group the main window adds
to its side column; ``CalibrationDialog`` implements the natural-feature
click calibration (no printed markers).  Everything reads/writes only
``StereoSettings`` / ``stereo_calibration.json`` - never the main
settings.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog,
                               QDoubleSpinBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QScrollArea, QSlider, QSpinBox,
                               QVBoxLayout)

from .calibration import (CalibrationError, StereoCalibration,
                          estimate_relative_pose)
from .capture import TimedCamera
from .config import CALIBRATION_PATH, StereoSettings


def _bgr_to_pixmap(frame: np.ndarray, width: int) -> QPixmap:
    h, w = frame.shape[:2]
    scale = width / w
    disp = cv2.resize(frame, (width, int(h * scale)),
                      interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
    img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1],
                 QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(img)


class _ClickLabel(QLabel):
    """Image label reporting clicks in *full-resolution* image coords."""

    def __init__(self, on_click) -> None:
        super().__init__()
        self._on_click = on_click
        self.scale = 1.0     # display px per image px

    def mousePressEvent(self, ev) -> None:  # noqa: N802 (Qt API)
        if ev.button() == Qt.MouseButton.LeftButton and self.scale > 0:
            x = ev.position().x() / self.scale
            y = ev.position().y() / self.scale
            self._on_click(x, y)
        super().mousePressEvent(ev)


class StereoPanel(QGroupBox):
    """Settings + status panel added to the main window (independent
    ON/OFF; default OFF; nothing here touches legacy settings)."""

    def __init__(self, worker, parent=None) -> None:
        super().__init__("2カメラ奥行き検出（ベータ）", parent)
        self.worker = worker
        self.s2: StereoSettings = worker.stereo_settings
        form = QFormLayout(self)

        self.enable_check = QCheckBox("2カメラ追跡を有効にする")
        self.enable_check.setChecked(self.s2.enabled)
        self.enable_check.setToolTip(
            "ONにするとカメラBと校正プロファイルを使って奥行きを推定します。\n"
            "OFF（初期値）では追加処理は一切動きません。トラッキング中でも"
            "切替できます。")
        self.enable_check.toggled.connect(self._set_enabled)
        form.addRow(self.enable_check)

        self.camb_combo = QComboBox()
        self._populate_cameras()
        self.camb_combo.currentIndexChanged.connect(self._apply)
        form.addRow("カメラB", self.camb_combo)

        row = QHBoxLayout()
        self.res_combo = QComboBox()
        for label in ("1280x720", "1920x1080", "960x540", "640x480"):
            self.res_combo.addItem(label)
        self.res_combo.setCurrentText(
            f"{self.s2.camera_b_width}x{self.s2.camera_b_height}")
        self.res_combo.currentIndexChanged.connect(self._apply)
        row.addWidget(self.res_combo)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(15, 120)
        self.fps_spin.setValue(self.s2.camera_b_fps)
        self.fps_spin.valueChanged.connect(self._apply)
        row.addWidget(QLabel("FPS"))
        row.addWidget(self.fps_spin)
        form.addRow("解像度", row)

        row = QHBoxLayout()
        self.fov_spin = QDoubleSpinBox()
        self.fov_spin.setRange(30.0, 175.0)
        self.fov_spin.setDecimals(1)
        self.fov_spin.setValue(self.s2.fov_deg)
        self.fov_spin.setSuffix(" °")
        self.fov_spin.valueChanged.connect(self._apply)
        row.addWidget(self.fov_spin)
        self.diag_check = QCheckBox("対角画角")
        self.diag_check.setChecked(self.s2.fov_is_diagonal)
        self.diag_check.setToolTip(
            "カメラ仕様の画角が対角表記の場合にチェック（通常は水平画角）")
        self.diag_check.toggled.connect(self._apply)
        row.addWidget(self.diag_check)
        form.addRow("画角", row)

        self.base_spin = QDoubleSpinBox()
        self.base_spin.setRange(0.05, 2.0)
        self.base_spin.setDecimals(3)
        self.base_spin.setSingleStep(0.01)
        self.base_spin.setValue(self.s2.baseline_m)
        self.base_spin.setSuffix(" m")
        self.base_spin.setToolTip("レンズ（光学中心）間の実測距離")
        self.base_spin.valueChanged.connect(self._apply)
        form.addRow("基線長", self.base_spin)

        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setRange(3.0, 50.0)
        self.tol_spin.setDecimals(0)
        self.tol_spin.setValue(self.s2.pair_tolerance_ms)
        self.tol_spin.setSuffix(" ms")
        self.tol_spin.setToolTip(
            "左右フレームをペアにする最大時刻差。小さいほど正確ですが"
            "ペア率が下がります（不足分は単眼で補完）")
        self.tol_spin.valueChanged.connect(self._apply)
        form.addRow("ペア許容時刻差", self.tol_spin)

        wrow = QHBoxLayout()
        self.weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.weight_slider.setRange(0, 100)
        self.weight_slider.setValue(int(self.s2.stereo_weight * 100))
        self.weight_label = QLabel(f"{int(self.s2.stereo_weight * 100)}%")
        self.weight_slider.valueChanged.connect(self._weight_changed)
        wrow.addWidget(self.weight_slider, 1)
        wrow.addWidget(self.weight_label)
        form.addRow("立体観測の反映度", wrow)

        self.depth_check = QCheckBox("前後移動を送信（腰の奥行き）")
        self.depth_check.setChecked(self.s2.send_depth_translation)
        self.depth_check.toggled.connect(self._apply)
        form.addRow(self.depth_check)
        row = QHBoxLayout()
        self.pip_check = QCheckBox("カメラB映像を小窓表示")
        self.pip_check.setChecked(self.s2.show_pip)
        self.pip_check.toggled.connect(self._apply)
        row.addWidget(self.pip_check)
        self.log_check = QCheckBox("観測ログを記録")
        self.log_check.setChecked(self.s2.debug_log)
        self.log_check.setToolTip(
            "stereo_logs/ にJSONL形式で観測・状態を記録します"
            "（検証・再現用。映像は保存しません）")
        self.log_check.toggled.connect(self._apply)
        row.addWidget(self.log_check)
        form.addRow(row)

        crow = QHBoxLayout()
        self.calib_label = QLabel()
        self.calib_label.setWordWrap(True)
        crow.addWidget(self.calib_label, 1)
        self.calib_btn = QPushButton("校正...")
        self.calib_btn.clicked.connect(self._open_calibration)
        crow.addWidget(self.calib_btn)
        form.addRow(crow)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color:#888;")
        form.addRow(self.status_label)
        self._refresh_calib_label()

    # ------------------------------------------------------------------
    def _populate_cameras(self) -> None:
        try:
            from ..camera import enumerate_cameras
            cams = enumerate_cameras()
        except Exception:
            cams = []
        if not cams:
            cams = [0, 1]
        for i in cams:
            self.camb_combo.addItem(f"カメラ {i}", i)
        idx = self.camb_combo.findData(self.s2.camera_b_index)
        if idx < 0:
            self.camb_combo.addItem(f"カメラ {self.s2.camera_b_index}",
                                    self.s2.camera_b_index)
            idx = self.camb_combo.count() - 1
        self.camb_combo.setCurrentIndex(idx)

    def _set_enabled(self, on: bool) -> None:
        self.s2.enabled = on
        self.worker.set_stereo_enabled(on)
        try:
            self.s2.save()      # persist the switch immediately
        except OSError:
            pass
        if on and StereoCalibration.load() is None:
            QMessageBox.information(
                self, "校正が必要です",
                "2カメラ追跡には校正プロファイルが必要です。\n"
                "トラッキング停止中に「校正...」から作成してください。\n"
                "校正が無い間は従来の1カメラ追跡で動作します。")

    def _weight_changed(self, v: int) -> None:
        self.weight_label.setText(f"{v}%")
        self.s2.stereo_weight = v / 100.0

    def _apply(self, *_a) -> None:
        s2 = self.s2
        s2.camera_b_index = self.camb_combo.currentData() or 0
        try:
            w, h = self.res_combo.currentText().split("x")
            s2.camera_b_width, s2.camera_b_height = int(w), int(h)
        except ValueError:
            pass
        s2.camera_b_fps = self.fps_spin.value()
        s2.fov_deg = self.fov_spin.value()
        s2.fov_is_diagonal = self.diag_check.isChecked()
        s2.baseline_m = self.base_spin.value()
        s2.pair_tolerance_ms = self.tol_spin.value()
        s2.send_depth_translation = self.depth_check.isChecked()
        s2.show_pip = self.pip_check.isChecked()
        s2.debug_log = self.log_check.isChecked()

    def save_settings(self) -> None:
        self._apply()
        try:
            self.s2.save()
        except OSError:
            pass

    def _refresh_calib_label(self) -> None:
        calib = StereoCalibration.load()
        if calib is None:
            self.calib_label.setText("校正: 未設定")
            self.calib_label.setStyleSheet("color:#c66;")
        else:
            status = {"approx": "近似校正", "verified": "検証済み"}.get(
                calib.status, calib.status)
            self.calib_label.setText(
                f"校正: {status}（{calib.created}／点{calib.n_inliers}"
                f"／残差{calib.residual_med_px:.1f}px"
                f"／基線{calib.baseline:.3f}m）")
            self.calib_label.setStyleSheet("color:#6a6;")

    def _open_calibration(self) -> None:
        if self.worker.get_status().running:
            QMessageBox.information(
                self, "確認",
                "校正はトラッキング停止中に行ってください"
                "（カメラを校正用に使用します）。")
            return
        self._apply()
        dlg = CalibrationDialog(
            self.worker.settings.camera_index,
            (self.worker.settings.camera_width,
             self.worker.settings.camera_height,
             self.worker.settings.camera_fps),
            self.s2, self)
        dlg.exec()
        self.save_settings()     # FOV / baseline used for the profile
        self._refresh_calib_label()

    def update_status(self, status) -> None:
        """Called from the main window's poll with TrackerStatus."""
        info = getattr(status, "stereo_info", "")
        if self.s2.enabled or info:
            self.status_label.setText(info or "待機中")
        else:
            self.status_label.setText("")


# ---------------------------------------------------------------------
class CalibrationDialog(QDialog):
    """Natural-feature click calibration.

    Flow: live preview -> capture stills -> click the same physical
    corners left/right (20-30 pairs, spread over the frame and depth) ->
    compute -> save.  A verification mode triangulates freshly clicked
    pairs so the user can compare distances with a tape measure.
    """

    _DISP_W = 560

    def __init__(self, cam_a_index: int, cam_a_mode: tuple,
                 s2: StereoSettings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("2カメラ校正（自然特徴クリック）")
        self.s2 = s2
        self._cam_a_index = cam_a_index
        self._frozen = False
        self._still_a: np.ndarray | None = None
        self._still_b: np.ndarray | None = None
        self._pts_a: list[list[float]] = []
        self._pts_b: list[list[float]] = []
        self._pending: tuple[float, float] | None = None  # clicked on A
        self._calib: StereoCalibration | None = StereoCalibration.load()
        self._verify_pts: list[np.ndarray] = []

        lay = QVBoxLayout(self)
        help_lbl = QLabel(
            "1) 両カメラを固定し「静止画を撮影」 2) 同じ物理的な角（机・棚"
            "・家具など）を 左→右 の順にクリック（20〜30組、画面全体と"
            "手前・奥に分散、同一平面に集中させない） 3)「校正を計算」→"
            "「保存」。マーカーや印刷物は不要です。")
        help_lbl.setWordWrap(True)
        lay.addWidget(help_lbl)

        imgs = QHBoxLayout()
        self.view_a = _ClickLabel(self._click_a)
        self.view_b = _ClickLabel(self._click_b)
        for v, title in ((self.view_a, "カメラA（メイン）"),
                         (self.view_b, "カメラB")):
            col = QVBoxLayout()
            col.addWidget(QLabel(title))
            scroll = QScrollArea()
            scroll.setWidget(v)
            scroll.setWidgetResizable(False)
            scroll.setMinimumSize(self._DISP_W + 20, 340)
            col.addWidget(scroll)
            imgs.addLayout(col)
        lay.addLayout(imgs)

        row = QHBoxLayout()
        self.freeze_btn = QPushButton("静止画を撮影")
        self.freeze_btn.clicked.connect(self._toggle_freeze)
        row.addWidget(self.freeze_btn)
        self.refine_check = QCheckBox("コーナー自動補正")
        self.refine_check.setChecked(True)
        self.refine_check.setToolTip(
            "クリック位置周辺でコーナーをサブピクセル精度に補正します")
        row.addWidget(self.refine_check)
        self.zoom_combo = QComboBox()
        for z in ("1x", "2x", "3x"):
            self.zoom_combo.addItem(z)
        self.zoom_combo.currentIndexChanged.connect(self._redraw)
        row.addWidget(QLabel("拡大"))
        row.addWidget(self.zoom_combo)
        self.verify_check = QCheckBox("検証モード")
        self.verify_check.setToolTip(
            "校正後にクリックした対応点の3D位置・距離を表示します。\n"
            "実測（メジャー等）と比較して精度を確認してください。")
        self.verify_check.toggled.connect(self._verify_toggled)
        row.addWidget(self.verify_check)
        row.addStretch(1)
        lay.addLayout(row)

        row = QHBoxLayout()
        self.count_label = QLabel("対応点: 0組")
        row.addWidget(self.count_label)
        undo_btn = QPushButton("最後の点を取消")
        undo_btn.clicked.connect(self._undo)
        row.addWidget(undo_btn)
        clear_btn = QPushButton("全て消去")
        clear_btn.clicked.connect(self._clear)
        row.addWidget(clear_btn)
        row.addStretch(1)
        self.compute_btn = QPushButton("校正を計算")
        self.compute_btn.clicked.connect(self._compute)
        row.addWidget(self.compute_btn)
        self.save_btn = QPushButton("保存")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save)
        row.addWidget(self.save_btn)
        self.verified_btn = QPushButton("検証済みとして保存")
        self.verified_btn.setEnabled(False)
        self.verified_btn.clicked.connect(self._save_verified)
        row.addWidget(self.verified_btn)
        close_btn = QPushButton("閉じる")
        close_btn.clicked.connect(self.reject)
        row.addWidget(close_btn)
        lay.addLayout(row)

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        lay.addWidget(self.result_label)

        # cameras
        self._cam_a = TimedCamera(cam_a_index, cam_a_mode[0], cam_a_mode[1],
                                  cam_a_mode[2])
        self._cam_b = TimedCamera(s2.camera_b_index, s2.camera_b_width,
                                  s2.camera_b_height, s2.camera_b_fps)
        errs = []
        if s2.camera_b_index == cam_a_index:
            errs.append("カメラAとカメラBに同じデバイスが選択されています。")
        else:
            if not self._cam_a.start():
                errs.append(self._cam_a.last_error or "カメラA起動失敗")
            if not self._cam_b.start():
                errs.append(self._cam_b.last_error or "カメラB起動失敗")
        if errs:
            QMessageBox.warning(self, "カメラエラー", "\n".join(errs))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(100)

    # -- capture / drawing --------------------------------------------
    def _tick(self) -> None:
        if self._frozen:
            return
        fa, _, _ = self._cam_a.latest()
        fb, _, _ = self._cam_b.latest()
        if fa is not None:
            self._still_a = fa.copy()
        if fb is not None:
            self._still_b = fb.copy()
        self._redraw()

    def _toggle_freeze(self) -> None:
        if not self._frozen and (self._still_a is None
                                 or self._still_b is None):
            QMessageBox.information(self, "確認",
                                    "両カメラの映像がまだ届いていません。")
            return
        self._frozen = not self._frozen
        self.freeze_btn.setText("再取得（ライブに戻す）" if self._frozen
                                else "静止画を撮影")
        self._redraw()

    def _zoom(self) -> int:
        return int(self.zoom_combo.currentText()[0])

    def _redraw(self) -> None:
        for view, still, pts in ((self.view_a, self._still_a, self._pts_a),
                                 (self.view_b, self._still_b, self._pts_b)):
            if still is None:
                continue
            img = still.copy()
            for i, (x, y) in enumerate(pts):
                cv2.drawMarker(img, (int(x), int(y)), (0, 255, 0),
                               cv2.MARKER_CROSS, 14, 2)
                cv2.putText(img, str(i + 1), (int(x) + 6, int(y) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            if view is self.view_a and self._pending is not None:
                x, y = self._pending
                cv2.drawMarker(img, (int(x), int(y)), (0, 200, 255),
                               cv2.MARKER_CROSS, 14, 2)
            width = self._DISP_W * self._zoom()
            pix = _bgr_to_pixmap(img, width)
            view.scale = width / img.shape[1]
            view.setPixmap(pix)
            view.resize(pix.size())
        n = len(self._pts_b)
        pend = "（右をクリック）" if self._pending is not None else ""
        self.count_label.setText(f"対応点: {n}組 {pend}")

    def _refine(self, img: np.ndarray, x: float, y: float
                ) -> tuple[float, float]:
        if not self.refine_check.isChecked():
            return x, y
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pts = np.array([[[x, y]]], dtype=np.float32)
        try:
            cv2.cornerSubPix(
                gray, pts, (6, 6), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                 20, 0.03))
            rx, ry = float(pts[0, 0, 0]), float(pts[0, 0, 1])
            if abs(rx - x) < 8 and abs(ry - y) < 8:
                return rx, ry
        except cv2.error:
            pass
        return x, y

    # -- clicking ------------------------------------------------------
    def _click_a(self, x: float, y: float) -> None:
        if not self._frozen or self._still_a is None:
            return
        x, y = self._refine(self._still_a, x, y)
        self._pending = (x, y)
        self._redraw()

    def _click_b(self, x: float, y: float) -> None:
        if not self._frozen or self._still_b is None:
            return
        if self._pending is None:
            QMessageBox.information(
                self, "確認", "先にカメラA側の対応点をクリックしてください。")
            return
        x, y = self._refine(self._still_b, x, y)
        if self.verify_check.isChecked():
            self._verify_pair(self._pending, (x, y))
            self._pending = None
            self._redraw()
            return
        self._pts_a.append(list(self._pending))
        self._pts_b.append([x, y])
        self._pending = None
        self._redraw()

    def _undo(self) -> None:
        if self._pending is not None:
            self._pending = None
        elif self._pts_b:
            self._pts_a.pop()
            self._pts_b.pop()
        self._redraw()

    def _clear(self) -> None:
        self._pts_a.clear()
        self._pts_b.clear()
        self._pending = None
        self._redraw()

    # -- calibration ---------------------------------------------------
    def _compute(self) -> None:
        if self._still_a is None or self._still_b is None:
            QMessageBox.information(self, "確認", "静止画を撮影してください。")
            return
        ha, wa = self._still_a.shape[:2]
        hb, wb = self._still_b.shape[:2]
        try:
            calib = estimate_relative_pose(
                np.array(self._pts_a), np.array(self._pts_b),
                (wa, ha), (wb, hb),
                self.s2.fov_deg, self.s2.fov_is_diagonal,
                self.s2.baseline_m,
                self._cam_a_index, self.s2.camera_b_index)
        except CalibrationError as e:
            self.result_label.setText(f"❌ {e}")
            self.result_label.setStyleSheet("color:#c66;")
            return
        self._calib = calib
        self.save_btn.setEnabled(True)
        self.result_label.setStyleSheet("color:#6a6;")
        self.result_label.setText(
            f"✅ 近似校正に成功: 有効点 {calib.n_inliers}/{calib.n_points}組, "
            f"再投影残差 中央値 {calib.residual_med_px:.2f}px / "
            f"p95 {calib.residual_p95_px:.2f}px, "
            f"基線 {calib.baseline:.3f}m。「保存」で適用されます。\n"
            "※近似校正です（公称画角・主点中央・歪みなしを仮定）。"
            "検証モードで実測と比較してください。")

    def _save(self) -> None:
        if self._calib is None:
            return
        self._calib.save(CALIBRATION_PATH)
        self.result_label.setText(
            self.result_label.text().split("\n")[0] + "\n保存しました: "
            + str(CALIBRATION_PATH))
        self.verified_btn.setEnabled(True)

    def _save_verified(self) -> None:
        if self._calib is None:
            return
        self._calib.status = "verified"
        self._calib.save(CALIBRATION_PATH)
        self.result_label.setText(
            "検証済みとして保存しました（検証は実施した距離・範囲での"
            "確認を意味します）。")

    # -- verification --------------------------------------------------
    def _verify_toggled(self, on: bool) -> None:
        if on and self._calib is None:
            QMessageBox.information(
                self, "確認", "先に校正を計算・保存してください。")
            self.verify_check.setChecked(False)
            return
        self._verify_pts.clear()
        if on:
            self.result_label.setText(
                "検証モード: 距離を確かめたい点を左→右の順にクリック。"
                "点の3D位置と、直前の点との距離を表示します。")

    def _verify_pair(self, pa, pb) -> None:
        X, err, valid = self._calib.triangulate(
            np.array([pa]), np.array([pb]), max_err_px=20.0)
        if not valid[0]:
            self.result_label.setText(
                f"⚠ この対応点は三角測量できません（残差 {err[0]:.1f}px）。"
                "クリック位置か左右の組を確認してください。")
            return
        p = X[0]
        self._verify_pts.append(p)
        txt = (f"点{len(self._verify_pts)}: カメラAから "
               f"X={p[0]:+.3f} Y={p[1]:+.3f} Z={p[2]:.3f} m "
               f"(残差 {err[0]:.1f}px)")
        if len(self._verify_pts) >= 2:
            d = float(np.linalg.norm(self._verify_pts[-1]
                                     - self._verify_pts[-2]))
            txt += f"\n直前の点との距離: {d:.3f} m（実測と比較してください）"
        self.result_label.setText(txt)

    # ------------------------------------------------------------------
    def done(self, r: int) -> None:  # noqa: A003
        self._timer.stop()
        self._cam_a.stop()
        self._cam_b.stop()
        super().done(r)
