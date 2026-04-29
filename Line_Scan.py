import datetime
import sys
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
    QRadioButton, QButtonGroup,
)

# --- How big is one pixel in real life? ---
# The camera sensor has tiny squares (pixels). Each square is 3 micrometers wide.
# But we're looking through a 275x microscope objective, so each pixel actually
# represents a much smaller piece of the sample.
# Real size per pixel = sensor pixel size / magnification
PIXEL_PITCH_UM = 3               # physical size of one camera pixel, in micrometers
OBJECTIVE_MAGNIFICATION = 275    # how much the objective lens magnifies the sample

PIXEL_SIZE_UM = PIXEL_PITCH_UM / OBJECTIVE_MAGNIFICATION  # real size of one pixel at the sample

# Turn on matplotlib's "interactive" mode so plots appear without freezing the program
plt.ion()

# Keeps track of the currently open plot window (None means no window is open)
_current_fig = None

# =============================================================================
# Gaussian fitting
# =============================================================================

def _gaussian(x, A, x0, w, B):
    # The mathematical shape of a laser beam is called a Gaussian (bell curve).
    # This formula returns the expected intensity at position x given:
    #   A  = peak height (how bright the center is)
    #   x0 = center position of the beam
    #   w  = beam radius (controls how wide the bell is) — the 1/e^2 half-width
    #   B  = background brightness (light even away from the beam)
    return A * np.exp(-2 * ((x - x0) ** 2) / (w ** 2)) + B

def fit_gaussian(x_vals, intensity_vals):
    # Tries to find the Gaussian curve that best matches the measured intensity data.
    # Returns a dictionary with the fit results, or {'success': False} if it failed.
    vals = np.array(intensity_vals, dtype=float)
    x    = np.array(x_vals, dtype=float)

    # Make an initial guess for each parameter so the fitting algorithm has a starting point
    B0   = float(np.min(vals))          # guess: background = dimmest value
    A0   = float(np.max(vals)) - B0    # guess: peak height = brightest minus background
    x0_0 = float(x[np.argmax(vals)])   # guess: center = position of the brightest pixel
    w0   = float(x[-1] - x[0]) / 8    # guess: beam width = 1/8 of the total scan range

    try:
        # curve_fit adjusts the parameters until the Gaussian matches the data as closely as possible
        popt, _ = curve_fit(_gaussian, x, vals, p0=[A0, x0_0, w0, B0], maxfev=5000)
        A, x0, w, B = popt
        span = x[-1] - x[0]

        # Sanity check: reject fits where the beam width is unrealistically tiny or huge
        if abs(w) < (span / 1000) or abs(w) > span:
            return {'success': False}
        return {'success': True, 'A': A, 'x0': x0, 'w': abs(w), 'B': B}
    except Exception:
        return {'success': False}  # fitting failed entirely (e.g. data too noisy)

# =============================================================================
# Beam profile plot  (opens as a separate matplotlib window)
# =============================================================================

def show_plot(x_mm, row, fit):
    global _current_fig

    # If a plot window is already open, close it cleanly before opening a new one.
    # This prevents crashes from trying to interact with a window the user already closed.
    if _current_fig is not None:
        try:
            plt.close(_current_fig)
        except Exception:
            pass
        _current_fig = None

    today = datetime.date.today()
    date_str = f"{today.month}/{today.day}/{today.year}"

    # Build the label that will appear at the bottom of the plot
    if fit['success']:
        diameter_um = 2 * fit['w'] * 1e3   # w is in mm; multiply by 2 for full diameter, x1000 for um
        diam_str = f"1/e^2 Gaussian Diameter = {diameter_um:.4f} um"
    else:
        diam_str = "Fit failed"

    # Create a new figure (the plot window) and remember it so we can close it safely later
    fig = plt.figure(figsize=(10, 5))
    _current_fig = fig

    # When the user closes this window, clear _current_fig so we know no window is open
    def _on_close(event):
        global _current_fig
        _current_fig = None

    fig.canvas.mpl_connect('close_event', _on_close)

    # Add the main plotting area — numbers set its position and size inside the figure
    ax = fig.add_axes([0.10, 0.28, 0.85, 0.65])

    # Plot the raw measured intensity as a blue line
    ax.plot(x_mm, row, color='blue', linewidth=1)

    if fit['success']:
        A, x0, w, B = fit['A'], fit['x0'], fit['w'], fit['B']

        # Draw the fitted Gaussian curve in red over the raw data
        x_fine = np.linspace(x_mm[0], x_mm[-1], 1000)  # smooth x-axis for the curve
        ax.plot(x_fine, _gaussian(x_fine, A, x0, w, B), color='red', linewidth=1.5)

        # Draw a dashed horizontal line at the 1/e^2 intensity level
        # (this is where the Gaussian drops to ~13.5% of its peak — the standard beam width definition)
        level = A / np.e ** 2 + B
        ax.axhline(level, color='red', linestyle='--', linewidth=1)

    ax.set_xlabel('Position (mm)', fontsize=10)
    ax.set_ylabel('Intensity (a.u.)', fontsize=10)
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))  # use scientific notation on axes
    ax.grid(True, alpha=0.4)
    ax.set_xlim(x_mm[0], x_mm[-1])

    # --- Footer section (mimics a POP beam profile report) ---
    fig.add_artist(mlines.Line2D([0.05, 0.95], [0.265, 0.265],
                                  transform=fig.transFigure, color='black', linewidth=0.8))
    fig.text(0.5, 0.215, 'POP beam profile', ha='center', va='top',
             fontsize=10, family='monospace')
    fig.add_artist(mlines.Line2D([0.05, 0.95], [0.175, 0.175],
                                  transform=fig.transFigure, color='black', linewidth=0.8))
    fig.text(0.05, 0.145, date_str, ha='left', va='top', fontsize=9, family='monospace')
    fig.text(0.05, 0.10,  diam_str, ha='left', va='top', fontsize=9, family='monospace')

    # Save the plot as an image file before displaying it
    plt.savefig('beam_profile.png', dpi=150, bbox_inches='tight')

    # Show the window without blocking the rest of the program.
    # Wrapped in try/except so that a closed or broken window doesn't crash the program.
    try:
        plt.show(block=False)
        plt.pause(0.001)  # tiny pause lets the GUI event loop process and actually display the window
    except Exception:
        pass

# =============================================================================
# Camera helpers
# =============================================================================

def select_pixel_format(camera):
    # Cameras can output images in different bit depths (more bits = more shades of grey).
    # We prefer 12-bit, then 10-bit, then 8-bit — whichever the camera supports.
    available = camera.get_pixel_formats()
    for fmt, depth in [(PixelFormat.Mono12, 12), (PixelFormat.Mono10, 10), (PixelFormat.Mono8, 8)]:
        if fmt in available:
            camera.set_pixel_format(fmt)
            return depth  # return how many bits per pixel we ended up with
    sys.exit("Error: no supported monochrome pixel format found on camera")

# =============================================================================
# Camera thread  — runs the capture loop in the background
# =============================================================================

class CameraThread(QThread):
    # Qt signal: fired once per frame with the raw numpy pixel array.
    # The GUI listens for this and updates the display each time it fires.
    frame_ready = Signal(object)
    connection_failed = Signal(str)   # emitted with an error message when startup fails

    def __init__(self):
        super().__init__()
        self._running = False
        self.bit_depth = 8
        self.max_val   = 255.0

    def run(self):
        # This method runs in a separate thread so the GUI never freezes waiting for a frame.
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
                        try:
                            frame = camera.get_frame(timeout_ms=2000)
                            img   = frame.as_numpy_ndarray()
                            self.frame_ready.emit(img)  # send frame to the GUI
                        except Exception:
                            break
        except Exception as exc:
            self.connection_failed.emit(f"Connection failed: {exc}")

    def stop(self):
        # Signal the loop to exit, then wait for the thread to finish cleanly.
        self._running = False
        self.wait()

# =============================================================================
# Main window
# =============================================================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Line Scanner")

        self._latest_frame  = None   # the most recent raw frame from the camera
        self._camera_thread = None   # CameraThread instance, or None when disconnected

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # Root layout splits the window into left controls and right camera feed
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
        panel.setFixedWidth(220)
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(8)

        # --- Line scan button ---
        # Disabled until the camera is connected; enabled in _on_camera_connected()
        self.scan_btn = QPushButton("LINE SCAN")
        self.scan_btn.setEnabled(False)
        self.scan_btn.setFixedHeight(40)
        self.scan_btn.clicked.connect(self._do_scan)
        layout.addWidget(self.scan_btn)

        layout.addSpacing(16)

        # --- Camera setting sliders ---
        self._all_mode_buttons = []   # all radio buttons; toggled on connect/disconnect
        for display_name, attr_prefix in [
            ("Exposure",  "exposure"),
            ("Gain",      "gain"),
            ("Intensity", "intensity"),
        ]:
            self._make_slider_group(display_name, attr_prefix, layout)

        layout.addStretch()
        return panel

    def _make_slider_group(self, display_name, attr_prefix, layout):
        # Header: setting name on the left, current value on the right
        header = QHBoxLayout()
        header.addWidget(QLabel(display_name))
        header.addStretch()
        val_lbl = QLabel("50")
        val_lbl.setAlignment(Qt.AlignRight)
        header.addWidget(val_lbl)
        layout.addLayout(header)

        # Mode row: Off / Once / Continuous radio buttons
        mode_row = QHBoxLayout()
        off_btn  = QRadioButton("Off")
        once_btn = QRadioButton("Once")
        cont_btn = QRadioButton("Continuous")
        off_btn.setChecked(True)
        for btn in (off_btn, once_btn, cont_btn):
            btn.setEnabled(False)
            mode_row.addWidget(btn)
            self._all_mode_buttons.append(btn)
        layout.addLayout(mode_row)

        group = QButtonGroup(self)
        group.addButton(off_btn,  0)
        group.addButton(once_btn, 1)
        group.addButton(cont_btn, 2)

        # Slider
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(50)
        slider.setEnabled(False)
        layout.addWidget(slider)
        layout.addSpacing(8)

        # Store refs under the expected attribute names
        setattr(self, f"{attr_prefix}_slider",      slider)
        setattr(self, f"{attr_prefix}_mode_group",  group)
        setattr(self, f"{attr_prefix}_value_label", val_lbl)
        setattr(self, f"{attr_prefix}_off_btn",     off_btn)

        # Value display
        slider.valueChanged.connect(lambda v: val_lbl.setText(str(v)))

        # Mode → slider enable/disable
        def on_mode(btn_id):
            slider.setEnabled(btn_id != 0)

        group.idClicked.connect(on_mode)

        # Once: revert to Off after the user releases the slider
        def on_released():
            if group.checkedId() == 1:
                off_btn.setChecked(True)
                slider.setEnabled(False)

        slider.sliderReleased.connect(on_released)

    def _make_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        # --- Connect / Disconnect button ---
        # Acts as a toggle: click once to start the camera, click again to stop it.
        self.connect_btn = QPushButton("CONNECT")
        self.connect_btn.setFixedHeight(36)
        self.connect_btn.setCheckable(True)   # stays "pressed" while camera is active
        self.connect_btn.clicked.connect(self._toggle_camera)
        layout.addWidget(self.connect_btn)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: red;")
        self.status_label.hide()
        layout.addWidget(self.status_label)

        # --- Camera feed display ---
        # Shows the live camera image as a scaled pixmap.
        # When the camera is off, this area stays black.
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black;")
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.camera_label)

        return panel

    def _make_divider(self):
        # Thin vertical line between the left and right panels
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    # ------------------------------------------------------------------
    # Camera control
    # ------------------------------------------------------------------

    def _toggle_camera(self, checked):
        if checked:
            # Start the camera
            try:
                self.connect_btn.setText("DISCONNECT")
                self.status_label.hide()
                self.status_label.setText("")
                self._camera_thread = CameraThread()
                self._camera_thread.frame_ready.connect(self._on_frame)
                self._camera_thread.connection_failed.connect(self._on_connection_failed)
                self._camera_thread.start()
                # Enable controls that need a live camera
                self.scan_btn.setEnabled(True)
                for btn in self._all_mode_buttons:
                    btn.setEnabled(True)
            except Exception as e:
                self.connect_btn.setChecked(False)
                self.connect_btn.setText("CONNECT")
                print(f"Error starting camera: {e}")
        else:
            # Stop the camera
            self.connect_btn.setText("CONNECT")
            if self._camera_thread:
                self._camera_thread.stop()
                self._camera_thread = None
            # Disable controls and clear the display
            self.scan_btn.setEnabled(False)
            for btn in self._all_mode_buttons:
                btn.setEnabled(False)
            for prefix in ("exposure", "gain", "intensity"):
                getattr(self, f"{prefix}_off_btn").setChecked(True)
                getattr(self, f"{prefix}_slider").setEnabled(False)
            self._latest_frame = None
            self.camera_label.clear()
            self.camera_label.setStyleSheet("background-color: black;")

    def _on_connection_failed(self, message: str):
        self.connect_btn.setChecked(False)
        self.connect_btn.setText("CONNECT")
        self.scan_btn.setEnabled(False)
        for btn in self._all_mode_buttons:
            btn.setEnabled(False)
        for prefix in ("exposure", "gain", "intensity"):
            getattr(self, f"{prefix}_off_btn").setChecked(True)
            getattr(self, f"{prefix}_slider").setEnabled(False)
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
        # Called automatically each time the camera thread emits a new frame.
        if self.status_label.isVisible():
            self.status_label.hide()
            self.status_label.setText("")
        self._latest_frame = img

        max_val = self._camera_thread.max_val if self._camera_thread else 255.0

        # Scale raw pixel values down to 8-bit (0-255) just for display;
        # the full-depth data is kept in self._latest_frame for the scan.
        display = (img.astype(float) / max_val * 255).astype(np.uint8)

        h, w = display.shape[:2]
        cx, cy = w // 2, h // 2

        # Convert grayscale to colour so we can draw a green crosshair
        display_bgr = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        cv2.line(display_bgr, (0, cy), (w, cy), (0, 255, 0), 2)   # horizontal scan line
        cv2.line(display_bgr, (cx, 0), (cx, h), (0, 255, 0), 2)   # vertical center line

        # Convert the OpenCV (BGR) array to a Qt image, then to a pixmap for the label
        rgb  = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Scale the pixmap to fit the label while keeping the camera's aspect ratio
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

        img    = self._latest_frame
        h, w   = img.shape[:2]
        cy     = h // 2

        # Extract the horizontal row of pixels running through the centre of the frame
        row  = img[cy].astype(float)

        # Convert pixel positions to millimetres, centred at zero
        x_mm = (np.arange(w) - w / 2) * PIXEL_SIZE_UM * 1e-3

        fit = fit_gaussian(x_mm, row)
        show_plot(x_mm, row, fit)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        # Make sure the camera thread shuts down when the window is closed
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
    window.resize(800, 500)
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
