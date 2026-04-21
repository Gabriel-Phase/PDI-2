import cv2
import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.ion()

scan_triggered = False

# --- Button ---

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

# --- Brightness ---

def pixel_to_brightness(bgr_pixel):
    b, g, r = int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])
    return int(0.299 * r + 0.587 * g + 0.114 * b)

# --- Gaussian fit ---

def _gaussian(x, A, x0, w, B):
    return A * np.exp(-2 * ((x - x0) ** 2) / (w ** 2)) + B

def fit_gaussian(x_vals, brightness_vals):
    vals = np.array(brightness_vals, dtype=float)
    x = np.array(x_vals, dtype=float)
    B0 = float(np.min(vals))
    A0 = float(np.max(vals)) - B0
    x0_0 = float(x[np.argmax(vals)])
    w0 = float(len(x)) / 8
    try:
        popt, _ = curve_fit(_gaussian, x, vals,
                            p0=[A0, x0_0, w0, B0],
                            maxfev=5000)
        A, x0, w, B = popt
        if abs(w) < 1 or abs(w) > len(x):
            return {'success': False}
        return {'success': True, 'A': A, 'x0': x0, 'w': abs(w), 'B': B}
    except Exception:
        return {'success': False}

# --- Plot ---

def show_plot(x_vals, brightness_vals, fit_result):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.array(x_vals)
    vals = np.array(brightness_vals)

    ax.plot(x, vals, color='blue', linewidth=1, label='Raw brightness')

    if fit_result['success']:
        A = fit_result['A']
        x0 = fit_result['x0']
        w = fit_result['w']
        B = fit_result['B']
        x_fine = np.linspace(x[0], x[-1], 1000)
        ax.plot(x_fine, _gaussian(x_fine, A, x0, w, B),
                color='red', linewidth=1.5, label='Gaussian fit')
        level = A / np.e ** 2 + B
        ax.axhline(level, color='red', linestyle='--', linewidth=1)
        ax.text(x[-1] * 0.55, level + 3,
                f'1/e\u00b2 Beam Diameter = {2 * w:.1f} px',
                color='red', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Fit failed', transform=ax.transAxes,
                ha='center', va='center', color='red', fontsize=12)

    ax.set_xlabel('Pixel Position (centered at 0)')
    ax.set_ylabel('Brightness')
    ax.set_title('Line Scan \u2014 Beam Profile')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

# --- Main loop ---

cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera")

btn_coords = (0, 0, 1, 1)
cv2.setMouseCallback("Camera", mouse_callback, {'btn': btn_coords})

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    if scan_triggered:
        scan_triggered = False
        scan_row = frame[cy]
        brightness = [pixel_to_brightness(p) for p in scan_row]
        x_vals = list(range(-(w // 2), w - (w // 2)))
        fit = fit_gaussian(x_vals, brightness)
        show_plot(x_vals, brightness, fit)

    cv2.line(frame, (0, cy), (w, cy), (0, 255, 0), 2)
    cv2.line(frame, (cx, 0), (cx, h), (0, 255, 0), 2)

    btn_coords = draw_button(frame)
    cv2.setMouseCallback("Camera", mouse_callback, {'btn': btn_coords})

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.close('all')
