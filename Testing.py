import cv2
import numpy as np

# Flag set to True when the user clicks the SCAN button
scan_triggered = False

# --- Button ---

def draw_button(frame):
    """Draw the SCAN button in the top-right corner and return its bounding box."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = w - 120, 10, w - 10, 50

    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)   # dark fill
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)      # green border
    cv2.putText(frame, "SCAN", (x1 + 18, y1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return (x1, y1, x2, y2)

def mouse_callback(event, x, y, flags, param):
    """Set scan_triggered=True when the user left-clicks inside the SCAN button."""
    global scan_triggered
    if event == cv2.EVENT_LBUTTONDOWN:
        bx1, by1, bx2, by2 = param['btn']
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            scan_triggered = True

# --- Brightness graph ---

def pixel_to_brightness(bgr_pixel):
    """Convert a BGR pixel to a single brightness value (standard luminance formula)."""
    b, g, r = int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])
    return int(0.299 * r + 0.587 * g + 0.114 * b)

def show_plot(brightness_values):
    """
    Draw a brightness-vs-pixel-position line graph and display it in its own window.
    brightness_values: list of ints in range [0, 255], one per pixel across the scan line.
    """
    # Canvas setup
    plot_h, plot_w = 380, 960
    margin = {'left': 90, 'right': 30, 'top': 30, 'bottom': 60}
    graph_w = plot_w - margin['left'] - margin['right']
    graph_h = plot_h - margin['top'] - margin['bottom']

    canvas = np.full((plot_h, plot_w, 3), 255, dtype=np.uint8)  # white background

    # Horizontal grid lines at brightness 0, 64, 128, 192, 255
    for val in range(0, 256, 64):
        gy = margin['top'] + graph_h - int((val / 255) * graph_h)
        cv2.line(canvas,
                 (margin['left'], gy), (plot_w - margin['right'], gy),
                 (60, 60, 60), 1)
        cv2.putText(canvas, str(val), (margin['left'] - 38, gy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

    # X and Y axes
    cv2.line(canvas,
             (margin['left'], margin['top']),
             (margin['left'], plot_h - margin['bottom']),
             (80, 80, 80), 1)
    cv2.line(canvas,
             (margin['left'], plot_h - margin['bottom']),
             (plot_w - margin['right'], plot_h - margin['bottom']),
             (80, 80, 80), 1)

    # Axis labels
    cv2.putText(canvas, "Brightness",
                (5, margin['top'] + graph_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
    cv2.putText(canvas, "Pixel Position (left -> right)",
                (margin['left'] + graph_w // 2 - 120, plot_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)

    # Convert brightness values to (x, y) pixel coordinates on the canvas
    n = len(brightness_values)
    points = []
    for i, val in enumerate(brightness_values):
        x = margin['left'] + int((i / (n - 1)) * graph_w)
        y = margin['top'] + graph_h - int((val / 255) * graph_h)
        points.append((x, y))

    # Draw the brightness curve
    for i in range(1, len(points)):
        cv2.line(canvas, points[i - 1], points[i], (255, 0, 0), 1)

    cv2.imshow("Line Scan - Brightness", canvas)

# --- Main loop ---

cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera")

# Placeholder coords — updated each frame after the button is drawn
btn_coords = (0, 0, 1, 1)
cv2.setMouseCallback("Camera", mouse_callback, {'btn': btn_coords})

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2  # center of the frame (crosshair position)

    # If the user clicked SCAN, read brightness along the horizontal center line
    if scan_triggered:
        scan_triggered = False
        scan_row = frame[cy]  # one row of BGR pixels
        brightness = [pixel_to_brightness(p) for p in scan_row]
        print(brightness)
        show_plot(brightness)

    # Draw crosshairs to show which row/column will be scanned
    cv2.line(frame, (0, cy), (w, cy), (0, 255, 0), 2)
    cv2.line(frame, (cx, 0), (cx, h), (0, 255, 0), 2)

    # Redraw button each frame (size/position can change if window is resized)
    btn_coords = draw_button(frame)
    cv2.setMouseCallback("Camera", mouse_callback, {'btn': btn_coords})

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
