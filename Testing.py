# --- IMPORTS ---
# cv2 (OpenCV): handles the camera feed and drawing on the video window
import cv2
# numpy: fast math on arrays of numbers (we use it for the brightness data)
import numpy as np
# matplotlib: creates the graph/plot that appears after a scan
import matplotlib.pyplot as plt
# curve_fit: scipy's tool that finds the best-fitting curve through a set of data points
from scipy.optimize import curve_fit

# Enable "interactive" plot mode so the graph window doesn't freeze the camera feed
plt.ion()

# Global flag — set to True when the user clicks SCAN, reset to False after the scan runs
scan_triggered = False


# --- SECTION: Draw the Scan Button ---

def draw_button(frame):
    """
    Draws a green SCAN button in the top-right corner of the camera frame.
    Returns the button's (x1, y1, x2, y2) corner coordinates so we can
    check later whether a mouse click landed inside it.
    """
    h, w = frame.shape[:2]  # get the frame's height and width in pixels

    # Calculate button position: 120px wide, 40px tall, 10px from the right/top edge
    x1, y1, x2, y2 = w - 120, 10, w - 10, 50

    # Draw a dark grey filled rectangle as the button background (-1 = filled)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)

    # Draw a green outline around the button (thickness = 2 pixels)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Write the "SCAN" label inside the button in green text
    cv2.putText(frame, "SCAN", (x1 + 18, y1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return (x1, y1, x2, y2)  # hand back the coordinates for click detection


def mouse_callback(event, x, y, flags, param):
    """
    OpenCV calls this function automatically every time the mouse does something
    in the camera window. We only care about left-button clicks.
    If the click position (x, y) is inside the SCAN button, we set the trigger flag.
    """
    global scan_triggered  # we need to modify the variable defined at the top of the file

    if event == cv2.EVENT_LBUTTONDOWN:  # only react to a left-button press
        bx1, by1, bx2, by2 = param['btn']  # unpack the button's corner coordinates
        # Check if the click landed inside the button's rectangular area
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            scan_triggered = True  # signal the main loop to run a scan


# --- SECTION: Convert Colour Pixel to Brightness ---

def pixel_to_brightness(bgr_pixel):
    """
    Takes a single pixel's colour values (Blue, Green, Red) and converts them
    to one brightness number between 0 (black) and 255 (white).
    The weights below match how human eyes perceive brightness — we're more
    sensitive to green than red, and least sensitive to blue.
    """
    b, g, r = int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])
    # Standard luminance formula: green is weighted most, blue the least
    return int(0.299 * r + 0.587 * g + 0.114 * b)


# --- SECTION: Gaussian Curve Model and Fitting ---

def _gaussian(x, A, x0, w, B):
    """
    The mathematical formula for a Gaussian (bell-curve) shape.
    This is the profile we expect a laser beam's brightness to follow.

    Parameters:
        x  — pixel position along the scan line
        A  — peak brightness above the background (how bright the centre is)
        x0 — centre position of the beam (where the peak sits)
        w  — 1/e² beam radius in pixels (controls how wide the bell curve is)
        B  — background brightness level (the flat floor under the curve)
    """
    return A * np.exp(-2 * ((x - x0) ** 2) / (w ** 2)) + B


def fit_gaussian(x_vals, brightness_vals):
    """
    Tries to find the Gaussian curve that best matches the measured brightness data.
    Uses scipy's curve_fit, which adjusts A, x0, w, and B until the curve fits.

    Returns a dict:
        {'success': True,  'A': ..., 'x0': ..., 'w': ..., 'B': ...}  on success
        {'success': False}  if the fit fails or the result looks unrealistic
    """
    vals = np.array(brightness_vals, dtype=float)
    x    = np.array(x_vals, dtype=float)

    # Initial guesses — curve_fit needs a starting point to search from
    B0   = float(np.min(vals))           # guess: background = darkest value
    A0   = float(np.max(vals)) - B0      # guess: peak height above background
    x0_0 = float(x[np.argmax(vals)])    # guess: centre is where the max brightness is
    w0   = float(len(x)) / 8            # guess: beam width is 1/8 of the frame width

    try:
        # curve_fit returns popt = [A, x0, w, B] that minimise the fit error
        # maxfev=5000 allows up to 5000 attempts before giving up
        popt, _ = curve_fit(_gaussian, x, vals,
                            p0=[A0, x0_0, w0, B0],
                            maxfev=5000)
        A, x0, w, B = popt

        # Sanity check: reject the fit if the beam width is unrealistically small or huge
        if abs(w) < 1 or abs(w) > len(x):
            return {'success': False}

        return {'success': True, 'A': A, 'x0': x0, 'w': abs(w), 'B': B}

    except Exception:
        # curve_fit throws an error if it can't converge — treat that as a failed fit
        return {'success': False}


# --- SECTION: Display the Scan Plot ---

def show_plot(x_vals, brightness_vals, fit_result):
    """
    Opens a matplotlib graph showing:
      - Blue line : the raw brightness values measured across the scan row
      - Red curve : the best-fit Gaussian (if the fit succeeded)
      - Red dashed line : the 1/e² brightness level (standard beam-size metric)
      - Text label : the calculated 1/e² beam diameter in pixels
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    x    = np.array(x_vals)
    vals = np.array(brightness_vals)

    # Plot the raw measured brightness data
    ax.plot(x, vals, color='blue', linewidth=1, label='Raw brightness')

    if fit_result['success']:
        A  = fit_result['A']   # peak amplitude
        x0 = fit_result['x0']  # centre position
        w  = fit_result['w']   # 1/e² radius
        B  = fit_result['B']   # background

        # Create 1000 evenly-spaced points for a smooth curve
        x_fine = np.linspace(x[0], x[-1], 1000)

        # Draw the fitted Gaussian curve in red
        ax.plot(x_fine, _gaussian(x_fine, A, x0, w, B),
                color='red', linewidth=1.5, label='Gaussian fit')

        # The 1/e² level is the brightness at which the beam intensity drops to 1/e² of its peak
        # 2w is the full 1/e² diameter (radius × 2)
        level = A / np.e ** 2 + B
        ax.axhline(level, color='red', linestyle='--', linewidth=1)

        # Annotate the diameter value on the plot
        ax.text(x[-1] * 0.55, level + 3,
                f'1/e\u00b2 Beam Diameter = {2 * w:.1f} px',
                color='red', fontsize=9)
    else:
        # If fitting failed, show a message in the centre of the plot
        ax.text(0.5, 0.5, 'Fit failed', transform=ax.transAxes,
                ha='center', va='center', color='red', fontsize=12)

    ax.set_xlabel('Pixel Position (centered at 0)')
    ax.set_ylabel('Brightness')
    ax.set_title('Line Scan \u2014 Beam Profile')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()

    # block=False means the plot opens without freezing the camera window
    plt.show(block=False)
    plt.pause(0.001)  # tiny pause lets matplotlib actually render the window


# --- SECTION: Main Camera Loop ---

# Open the default webcam (0 = first available camera)
cap = cv2.VideoCapture(0)

# Create the display window
cv2.namedWindow("Camera")

# Placeholder coordinates for the button — updated every frame inside the loop
btn_coords = (0, 0, 1, 1)
cv2.setMouseCallback("Camera", mouse_callback, {'btn': btn_coords})

while True:
    ret, frame = cap.read()  # capture one frame from the camera
    if not ret:
        break  # stop if the camera feed drops

    h, w = frame.shape[:2]      # frame dimensions in pixels
    cx, cy = w // 2, h // 2    # pixel coordinates of the frame's centre

    # --- Handle a scan trigger ---
    if scan_triggered:
        scan_triggered = False  # reset the flag immediately so we don't scan twice

        # Extract the horizontal row of pixels that runs through the centre of the frame
        scan_row = frame[cy]

        # Convert every pixel in that row to a single brightness value
        brightness = [pixel_to_brightness(p) for p in scan_row]

        # Build x-axis values centered at 0 (e.g. -320 to 319 for a 640px wide frame)
        x_vals = list(range(-(w // 2), w - (w // 2)))

        # Fit a Gaussian to the brightness profile and show the result
        fit = fit_gaussian(x_vals, brightness)
        show_plot(x_vals, brightness, fit)

    # Draw green crosshairs at the centre so the user can see the scan line
    cv2.line(frame, (0, cy), (w, cy), (0, 255, 0), 2)   # horizontal line
    cv2.line(frame, (cx, 0), (cx, h), (0, 255, 0), 2)   # vertical line

    # Redraw the button and update its coordinates for this frame
    btn_coords = draw_button(frame)
    cv2.setMouseCallback("Camera", mouse_callback, {'btn': btn_coords})

    # Show the current frame in the window
    cv2.imshow("Camera", frame)

    # Wait 1 ms for a keypress; if the user presses 'q', quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()           # release the webcam so other apps can use it
cv2.destroyAllWindows() # close all OpenCV windows
plt.close('all')        # close any open matplotlib plot windows
