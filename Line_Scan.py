import datetime
import sys

import cv2
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from vmbpy import VmbSystem, PixelFormat

# Physical pixel size at the sample plane.
# Set this to: camera_sensor_pixel_pitch_um / objective_magnification
# Example: 4.4 um pixel pitch / 50x objective = 0.088 um/pixel
PIXEL_SIZE_UM = 0.088

plt.ion()

scan_triggered = False


# --- Button and mouse ---

def draw_button(frame):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = w - 120, 10, w - 10, 50
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "SCAN", (x1 + 18, y1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return (x1, y1, x2, y2)


def mouse_callback(event, x, y, flags, param):
    global scan_triggered
    if event == cv2.EVENT_LBUTTONDOWN:
        bx1, by1, bx2, by2 = param['btn']
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            scan_triggered = True


# --- Gaussian fitting ---

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


# --- Plot (POP beam profile format) ---

def show_plot(x_mm, row, fit):
    today = datetime.date.today()
    date_str = f"{today.month}/{today.day}/{today.year}"

    if fit['success']:
        diameter_um = 2 * fit['w'] * 1e3
        diam_str = f"1/e^2 Gaussian Diameter = {diameter_um:.4f} um"
    else:
        diam_str = "Fit failed"

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_axes([0.10, 0.28, 0.85, 0.65])

    ax.plot(x_mm, row, color='blue', linewidth=1)

    if fit['success']:
        A, x0, w, B = fit['A'], fit['x0'], fit['w'], fit['B']
        x_fine = np.linspace(x_mm[0], x_mm[-1], 1000)
        ax.plot(x_fine, _gaussian(x_fine, A, x0, w, B), color='red', linewidth=1.5)
        level = A / np.e ** 2 + B
        ax.axhline(level, color='red', linestyle='--', linewidth=1)

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
    plt.show(block=False)
    plt.pause(0.001)


# --- Pixel format selection ---

def select_pixel_format(camera):
    available = camera.get_pixel_formats()
    for fmt, depth in [(PixelFormat.Mono12, 12), (PixelFormat.Mono10, 10), (PixelFormat.Mono8, 8)]:
        if fmt in available:
            camera.set_pixel_format(fmt)
            return depth
    sys.exit("Error: no supported monochrome pixel format found on camera")


# --- Main ---

def main():
    with VmbSystem.get_instance() as vmb:
        cameras = vmb.get_all_cameras()
        if not cameras:
            sys.exit("Error: no Allied Vision cameras found")

        with cameras[0] as camera:
            bit_depth = select_pixel_format(camera)
            max_val   = float(2 ** bit_depth - 1)

            cv2.namedWindow("GoldenEye")
            btn_coords = (0, 0, 1, 1)
            cv2.setMouseCallback("GoldenEye", mouse_callback, {'btn': btn_coords})

            while True:
                global scan_triggered
                frame = camera.get_frame(timeout_ms=2000)
                img   = frame.as_numpy_ndarray()        # shape (h, w), full bit-depth
                h, w  = img.shape[:2]
                cx, cy = w // 2, h // 2

                # Normalize to 8-bit for display
                display = (img.astype(float) / max_val * 255).astype(np.uint8)
                display_bgr = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

                if scan_triggered:
                    scan_triggered = False
                    row   = img[cy].astype(float)
                    x_mm  = (np.arange(w) - w / 2) * PIXEL_SIZE_UM * 1e-3
                    fit   = fit_gaussian(x_mm, row)
                    show_plot(x_mm, row, fit)

                cv2.line(display_bgr, (0, cy), (w, cy), (0, 255, 0), 2)
                cv2.line(display_bgr, (cx, 0), (cx, h), (0, 255, 0), 2)

                btn_coords = draw_button(display_bgr)
                cv2.setMouseCallback("GoldenEye", mouse_callback, {'btn': btn_coords})
                cv2.imshow("GoldenEye", display_bgr)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()
    plt.close('all')


if __name__ == '__main__':
    main()
