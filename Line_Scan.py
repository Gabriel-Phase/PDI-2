import datetime
import queue
import sys
import time
import cv2
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from vmbpy import VmbSystem, PixelFormat

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QPushButton,
    QLabel, QSlider, QSizePolicy, QFrame,
    QLineEdit,
)

PIXEL_PITCH_UM  = 5.0   # Alvium G1-030 VSWIR sensor pixel pitch (µm)
OBJECTIVE_MAG   = 100   # Mitutoyo 100x M Plan APO NIR
ZOOM_MAG        = 5.25  # Optem Fusion 7:1 at maximum zoom (range: 0.75x–5.25x)
CAMERA_TUBE_MAG = 1.0   # Optem Fusion camera tube 35-08-06-000

PIXEL_SIZE_UM = PIXEL_PITCH_UM / (OBJECTIVE_MAG * ZOOM_MAG * CAMERA_TUBE_MAG)

# Turn on matplotlib's "interactive" mode so plots appear without freezing the program
plt.ion()

# Keeps track of the currently open plot window (None means no window is open)
_current_fig = None

# =============================================================================
# Gaussian fitting
# =============================================================================

def _gaussian(x, A, x0, w, B):
    return A * np.exp(-2 * ((x - x0) ** 2) / (w ** 2)) + B

def fit_gaussian(x_vals, intensity_vals):
    vals = np.array(intensity_vals, dtype=float)
    x    = np.array(x_vals, dtype=float)

    B0   = float(np.min(vals))
    A0   = float(np.max(vals)) - B0
    x0_0 = float(x[np.argmax(vals)])
    w0   = float(x[-1] - x[0]) / 8

    try:
        popt, _ = curve_fit(_gaussian, x, vals, p0=[A0, x0_0, w0, B0], maxfev=5000)
        A, x0, w, B = popt
        span = x[-1] - x[0]

        if abs(w) < (span / 1000) or abs(w) > span:
            return {'success': False}
        return {'success': True, 'A': A, 'x0': x0, 'w': abs(w), 'B': B}
    except Exception:
        return {'success': False}

# =============================================================================
# Beam profile plot  (opens as a separate matplotlib window)
# =============================================================================

def show_plot(x_mm, row, fit):
    global _current_fig

    if _current_fig is not None:
        try:
            plt.close(_current_fig)
        except Exception:
            pass
        _current_fig = None

    today = datetime.date.today()
    date_str = f"{today.month}/{today.day}/{today.year}"

    if fit['success']:
        diameter_um = 2 * fit['w'] * 1e3
        diam_str = f"1/e^2 Gaussian Diameter = {diameter_um:.4f} um"
    else:
        diam_str = "Fit failed"

    fig = plt.figure(figsize=(10, 5))
    _current_fig = fig

    def _on_close(event):
        global _current_fig
        _current_fig = None

    fig.canvas.mpl_connect('close_event', _on_close)

    ax = fig.add_axes([0.10, 0.28, 0.85, 0.65])

    ax.plot(x_mm, row, color='blue', linewidth=1)

    if fit['success']:
        A, x0, w, B = fit['A'], fit['x0'], fit['w'], fit['B']

        x_fine = np.linspace(x_mm[0], x_mm[-1], 1000)
        ax.plot(x_fine, _gaussian(x_fine, A, x0, w, B), color='red', linewidth=1.5)

        level = A / np.e ** 2 + B
        ax.axhline(level, color='red', linestyle='--', linewidth=1)

    ax.set_xlabel('Position (µm)', fontsize=10)
    ax.set_ylabel('Intensity (a.u.)', fontsize=10)
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.grid(True, alpha=0.4)
    ax.set_xlim(x_mm[0], x_mm[-1])

    fig.add_artist(mlines.Line2D([0.05, 0.95], [0.265, 0.265],
                                  transform=fig.transFigure, color='black', linewidth=0.8))
    fig.text(0.5, 0.215, 'POP beam profile', ha='center', va='top',
             fontsize=10, family='monospace')
    fig.add_artist(mlines.Line2D([0.05, 0.95], [0.175, 0.175],
                                  transform=fig.transFigure, color='black', linewidth=0.8))
    fig.text(0.05, 0.145, date_str, ha='left', va='top', fontsize=9, family='monospace')
    fig.text(0.05, 0.10,  diam_str, ha='left', va='top', fontsize=9, family='monospace')

    plt.savefig('beam_profile.png', dpi=150, bbox_inches='tight')

    try:
        plt.show(block=False)
        plt.pause(0.001)
    except Exception:
        pass

# =============================================================================
# Camera helpers
# =============================================================================

def select_pixel_format(camera):
    available = camera.get_pixel_formats()
    for fmt, depth in [(PixelFormat.Mono12, 12), (PixelFormat.Mono10, 10), (PixelFormat.Mono8, 8)]:
        if fmt in available:
            camera.set_pixel_format(fmt)
            return depth
    sys.exit("Error: no supported monochrome pixel format found on camera")

# Logarithmic mapping helpers for exposure (100–1,000,000 µs over slider range 0–1000)
_EXP_MIN_LOG = 2.0        # log10(100)
_EXP_MAX_LOG = 6.0        # log10(1,000,000)

def _slider_to_exposure(pos):
    return 10 ** (_EXP_MIN_LOG + pos / 1000.0 * (_EXP_MAX_LOG - _EXP_MIN_LOG))

def _exposure_to_slider(us):
    us = max(100.0, min(1_000_000.0, float(us)))
    return round((np.log10(us) - _EXP_MIN_LOG) / (_EXP_MAX_LOG - _EXP_MIN_LOG) * 1000)

# =============================================================================
# Camera thread  — runs the capture loop in the background
# =============================================================================

class CameraThread(QThread):
    frame_ready = Signal(object)
    connection_failed = Signal(str)

    def __init__(self):
        super().__init__()
        self._running  = False
        self.bit_depth = 8
        self.max_val   = 255.0
        self._cmd_queue = queue.Queue()

    def send_command(self, cmd: dict):
        self._cmd_queue.put_nowait(cmd)

    def run(self):
        self._running = True
        try:
            with VmbSystem.get_instance() as vmb:
                cameras = vmb.get_all_cameras()
                if not cameras:
                    self.connection_failed.emit("Connection failed: no cameras found.")
                    return
                with cameras[0] as camera:
                    self.bit_depth = select_pixel_format(camera)
                    self.max_val   = float(2 ** self.bit_depth - 1)
                    while self._running:
                        # Apply any pending camera commands before grabbing a frame
                        while not self._cmd_queue.empty():
                            cmd = self._cmd_queue.get_nowait()
                            try:
                                if 'exposure' in cmd:
                                    camera.get_feature_by_name('ExposureTime').set(float(cmd['exposure']))
                                if 'gain' in cmd:
                                    camera.get_feature_by_name('Gain').set(float(cmd['gain']))
                                if 'exposure_auto' in cmd:
                                    camera.get_feature_by_name('ExposureAuto').set(cmd['exposure_auto'])
                                if 'gain_auto' in cmd:
                                    camera.get_feature_by_name('GainAuto').set(cmd['gain_auto'])
                            except Exception:
                                pass

                        try:
                            frame = camera.get_frame(timeout_ms=2000)
                            img   = frame.as_numpy_ndarray()
                            self.frame_ready.emit(img)
                        except Exception:
                            break
        except Exception as exc:
            self.connection_failed.emit(f"Connection failed: {exc}")

    def stop(self):
        self._running = False
        self.wait()

# =============================================================================
# Main window
# =============================================================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Line Scanner")

        self._latest_frame  = None
        self._camera_thread = None
        self._last_frame_time = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setSpacing(12)
        root.setContentsMargins(12, 12, 12, 12)

        root.addWidget(self._make_left_panel())
        root.addWidget(self._make_divider())
        root.addWidget(self._make_right_panel(), stretch=1)

    # ------------------------------------------------------------------
    # Panel builders
    # ------------------------------------------------------------------

    def _make_left_panel(self):
        panel = QWidget()
        panel.setFixedWidth(380)
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(8)

        self.scan_btn = QPushButton("LINE SCAN")
        self.scan_btn.setEnabled(False)
        self.scan_btn.setFixedHeight(40)
        self.scan_btn.clicked.connect(self._do_scan)
        layout.addWidget(self.scan_btn)

        layout.addSpacing(16)

        self._make_exposure_group(layout)
        layout.addSpacing(8)
        self._make_gain_group(layout)

        layout.addStretch()
        return panel

    def _make_exposure_group(self, layout):
        layout.addWidget(QLabel("Exposure"))

        # Slider + value box + unit
        row = QHBoxLayout()
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(0, 1000)
        self.exposure_slider.setValue(_exposure_to_slider(468.0))
        self.exposure_slider.setEnabled(False)
        row.addWidget(self.exposure_slider)

        self.exposure_lineedit = QLineEdit("468.4")
        self.exposure_lineedit.setFixedWidth(72)
        self.exposure_lineedit.setEnabled(False)
        row.addWidget(self.exposure_lineedit)

        row.addWidget(QLabel("[µs]"))
        layout.addLayout(row)

        # Auto label + buttons
        layout.addWidget(QLabel("Exposure Auto"))
        auto_row = QHBoxLayout()
        self.exposure_auto_buttons = []
        for label in ("Off", "Once", "Continuous"):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setEnabled(False)
            auto_row.addWidget(btn)
            self.exposure_auto_buttons.append(btn)
        self.exposure_auto_buttons[0].setChecked(True)  # Off selected by default
        layout.addLayout(auto_row)

        # Wire up slider ↔ lineedit
        self._updating_exposure = False

        def on_exposure_slider(pos):
            if self._updating_exposure:
                return
            us = _slider_to_exposure(pos)
            self._updating_exposure = True
            self.exposure_lineedit.setText(f"{us:.1f}")
            self._updating_exposure = False
            if self._camera_thread:
                self._camera_thread.send_command({'exposure': us})

        def on_exposure_edit():
            if self._updating_exposure:
                return
            try:
                us = float(self.exposure_lineedit.text())
            except ValueError:
                return
            us = max(100.0, min(1_000_000.0, us))
            self._updating_exposure = True
            self.exposure_lineedit.setText(f"{us:.1f}")
            self.exposure_slider.setValue(_exposure_to_slider(us))
            self._updating_exposure = False
            if self._camera_thread:
                self._camera_thread.send_command({'exposure': us})

        self.exposure_slider.valueChanged.connect(on_exposure_slider)
        self.exposure_lineedit.editingFinished.connect(on_exposure_edit)

        # Wire up auto buttons (mutually exclusive toggle)
        auto_modes = ('Off', 'Once', 'Continuous')

        def make_auto_handler(idx, mode):
            def handler(checked):
                if not checked:
                    return
                for i, b in enumerate(self.exposure_auto_buttons):
                    b.setChecked(i == idx)
                manual_enabled = (mode == 'Off')
                self.exposure_slider.setEnabled(manual_enabled)
                self.exposure_lineedit.setEnabled(manual_enabled)
                if self._camera_thread:
                    self._camera_thread.send_command({'exposure_auto': mode})
                if mode == 'Once':
                    # Revert to Off after one auto-adjust cycle
                    self.exposure_auto_buttons[0].setChecked(True)
                    self.exposure_auto_buttons[idx].setChecked(False)
                    self.exposure_slider.setEnabled(True)
                    self.exposure_lineedit.setEnabled(True)
            return handler

        for i, (btn, mode) in enumerate(zip(self.exposure_auto_buttons, auto_modes)):
            btn.clicked.connect(make_auto_handler(i, mode))

    def _make_gain_group(self, layout):
        layout.addWidget(QLabel("Gain"))

        row = QHBoxLayout()
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(0, 400)   # 0.1 dB steps → 0–40 dB
        self.gain_slider.setValue(0)
        self.gain_slider.setEnabled(False)
        row.addWidget(self.gain_slider)

        self.gain_lineedit = QLineEdit("0.0")
        self.gain_lineedit.setFixedWidth(72)
        self.gain_lineedit.setEnabled(False)
        row.addWidget(self.gain_lineedit)

        row.addWidget(QLabel("[dB]"))
        layout.addLayout(row)

        layout.addWidget(QLabel("Gain Auto"))
        auto_row = QHBoxLayout()
        self.gain_auto_buttons = []
        for label in ("Off", "Once", "Continuous"):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setEnabled(False)
            auto_row.addWidget(btn)
            self.gain_auto_buttons.append(btn)
        self.gain_auto_buttons[0].setChecked(True)
        layout.addLayout(auto_row)

        self._updating_gain = False

        def on_gain_slider(pos):
            if self._updating_gain:
                return
            db = pos / 10.0
            self._updating_gain = True
            self.gain_lineedit.setText(f"{db:.1f}")
            self._updating_gain = False
            if self._camera_thread:
                self._camera_thread.send_command({'gain': db})

        def on_gain_edit():
            if self._updating_gain:
                return
            try:
                db = float(self.gain_lineedit.text())
            except ValueError:
                return
            db = max(0.0, min(40.0, db))
            self._updating_gain = True
            self.gain_lineedit.setText(f"{db:.1f}")
            self.gain_slider.setValue(round(db * 10))
            self._updating_gain = False
            if self._camera_thread:
                self._camera_thread.send_command({'gain': db})

        self.gain_slider.valueChanged.connect(on_gain_slider)
        self.gain_lineedit.editingFinished.connect(on_gain_edit)

        auto_modes = ('Off', 'Once', 'Continuous')

        def make_gain_auto_handler(idx, mode):
            def handler(checked):
                if not checked:
                    return
                for i, b in enumerate(self.gain_auto_buttons):
                    b.setChecked(i == idx)
                manual_enabled = (mode == 'Off')
                self.gain_slider.setEnabled(manual_enabled)
                self.gain_lineedit.setEnabled(manual_enabled)
                if self._camera_thread:
                    self._camera_thread.send_command({'gain_auto': mode})
                if mode == 'Once':
                    self.gain_auto_buttons[0].setChecked(True)
                    self.gain_auto_buttons[idx].setChecked(False)
                    self.gain_slider.setEnabled(True)
                    self.gain_lineedit.setEnabled(True)
            return handler

        for i, (btn, mode) in enumerate(zip(self.gain_auto_buttons, auto_modes)):
            btn.clicked.connect(make_gain_auto_handler(i, mode))

    def _make_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        self.connect_btn = QPushButton("CONNECT")
        self.connect_btn.setFixedHeight(36)
        self.connect_btn.setCheckable(True)
        self.connect_btn.clicked.connect(self._toggle_camera)
        layout.addWidget(self.connect_btn)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: red;")
        self.status_label.hide()
        layout.addWidget(self.status_label)

        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black;")
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.camera_label)

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setAlignment(Qt.AlignRight)
        self.fps_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self.fps_label)

        return panel

    def _make_divider(self):
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    # ------------------------------------------------------------------
    # Camera control
    # ------------------------------------------------------------------

    def _set_controls_enabled(self, enabled: bool):
        self.scan_btn.setEnabled(enabled)
        self.exposure_slider.setEnabled(enabled)
        self.exposure_lineedit.setEnabled(enabled)
        self.gain_slider.setEnabled(enabled)
        self.gain_lineedit.setEnabled(enabled)
        for btn in self.exposure_auto_buttons:
            btn.setEnabled(enabled)
        for btn in self.gain_auto_buttons:
            btn.setEnabled(enabled)
        if not enabled:
            self.exposure_auto_buttons[0].setChecked(True)
            self.gain_auto_buttons[0].setChecked(True)

    def _toggle_camera(self, checked):
        if checked:
            try:
                self.connect_btn.setText("DISCONNECT")
                self.status_label.hide()
                self.status_label.setText("")
                self._camera_thread = CameraThread()
                self._camera_thread.frame_ready.connect(self._on_frame)
                self._camera_thread.connection_failed.connect(self._on_connection_failed)
                self._camera_thread.start()
                self._set_controls_enabled(True)
            except Exception as e:
                self.connect_btn.setChecked(False)
                self.connect_btn.setText("CONNECT")
                print(f"Error starting camera: {e}")
        else:
            self.connect_btn.setText("CONNECT")
            if self._camera_thread:
                self._camera_thread.stop()
                self._camera_thread = None
            self._set_controls_enabled(False)
            self._latest_frame = None
            self._last_frame_time = None
            self.camera_label.clear()
            self.camera_label.setStyleSheet("background-color: black;")
            self.fps_label.setText("FPS: --")

    def _on_connection_failed(self, message: str):
        self.connect_btn.setChecked(False)
        self.connect_btn.setText("CONNECT")
        self._set_controls_enabled(False)
        self.status_label.setText(message)
        self.status_label.show()
        if self._camera_thread:
            self._camera_thread.quit()
            self._camera_thread.wait()
            self._camera_thread = None

    # ------------------------------------------------------------------
    # Frame display
    # ------------------------------------------------------------------

    def _on_frame(self, img):
        if self.status_label.isVisible():
            self.status_label.hide()
            self.status_label.setText("")
        self._latest_frame = img

        now = time.monotonic()
        if self._last_frame_time is not None:
            fps = 1.0 / (now - self._last_frame_time)
            self.fps_label.setText(f"FPS: {fps:.1f}")
        self._last_frame_time = now

        max_val = self._camera_thread.max_val if self._camera_thread else 255.0

        display = (img.astype(float) / max_val * 255).astype(np.uint8)

        h, w = display.shape[:2]
        cx, cy = w // 2, h // 2

        display_bgr = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        cv2.line(display_bgr, (0, cy), (w, cy), (0, 255, 0), 2)
        cv2.line(display_bgr, (cx, 0), (cx, h), (0, 255, 0), 2)

        rgb  = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        scaled = pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.camera_label.setPixmap(scaled)

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    def _do_scan(self):
        if self._latest_frame is None:
            return

        img  = self._latest_frame
        h, w = img.shape[:2]
        cy   = h // 2

        # ---------------------------------------------------------
        # FIX 1: Row Averaging for Noise Reduction
        # ---------------------------------------------------------
        # Safely grab 11 rows around the center (handling edge cases if image is tiny)
        top_row = max(0, cy - 5)
        bot_row = min(h, cy + 6)
        
        # Average the rows together to smooth out digital noise
        row = np.mean(img[top_row:bot_row], axis=0).astype(float)
        
        # ---------------------------------------------------------
        # FIX 2: Beam Clipping Detection
        # ---------------------------------------------------------
        peak_intensity = np.max(row)
        baseline_estimate = np.min(row)
        amplitude = peak_intensity - baseline_estimate

        # Check how "bright" the edges of the sensor are relative to the peak
        left_edge = row[0] - baseline_estimate
        right_edge = row[-1] - baseline_estimate
        clip_threshold = 0.15 * amplitude  # 15% threshold

        # Warn the UI if the beam hasn't tapered off by the edges of the sensor
        if left_edge > clip_threshold or right_edge > clip_threshold:
            self.status_label.setText("WARNING: Beam is clipped! Please zoom out or adjust alignment.")
            self.status_label.show()
        else:
            # Clear the warning if the beam is contained safely within the frame
            self.status_label.hide()
            self.status_label.setText("")

        # ---------------------------------------------------------
        # Execute Math & Plotting
        # ---------------------------------------------------------
        # Calculate physical x-axis in millimeters based on your correct optical math
        x_mm = (np.arange(w) - w / 2) * PIXEL_SIZE_UM * 1e-3
        
        # Fit the Gaussian and display
        fit = fit_gaussian(x_mm, row)
        show_plot(x_mm, row, fit)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._camera_thread:
            self._camera_thread.stop()
        plt.close('all')
        event.accept()

# =============================================================================
# Entry point
# =============================================================================

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1000, 500)
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
