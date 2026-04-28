import datetime
import sys
import cv2                          # OpenCV: used to show the live camera window
import matplotlib.lines as mlines  # lets us draw straight lines on the plot
import matplotlib.pyplot as plt     # used to draw the beam profile graph
import numpy as np                  # math library for arrays and number crunching
from scipy.optimize import curve_fit  # finds the best-fit curve through data points
from vmbpy import VmbSystem, PixelFormat  # Allied Vision camera driver

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

# This flag becomes True when the user clicks the SCAN button
scan_triggered = False

# Keeps track of the currently open plot window (None means no window is open)
_current_fig = None

# --- Button and mouse ---

def draw_button(frame):
    # Draws a green "SCAN" button in the top-right corner of the camera window.
    # Returns the button's corner coordinates so we know where to detect clicks.
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = w - 120, 10, w - 10, 50  # pixel coordinates of the button box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)   # dark grey fill
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)     # green border
    cv2.putText(frame, "SCAN", (x1 + 18, y1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)    # green label
    return (x1, y1, x2, y2)

def mouse_callback(event, x, y, flags, param):
    # OpenCV calls this function automatically every time the user clicks in the window.
    # We check if the click landed inside the SCAN button; if so, set the trigger flag.
    global scan_triggered
    if event == cv2.EVENT_LBUTTONDOWN:          # left mouse button was pressed
        bx1, by1, bx2, by2 = param['btn']       # unpack button corners from param dict
        if bx1 <= x <= bx2 and by1 <= y <= by2: # was the click inside the button?
            scan_triggered = True

# --- Gaussian fitting ---

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

# --- Plot (POP beam profile format) ---

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
    # Horizontal divider line below the plot
    fig.add_artist(mlines.Line2D([0.05, 0.95], [0.265, 0.265],
                                  transform=fig.transFigure, color='black', linewidth=0.8))
    fig.text(0.5, 0.215, 'POP beam profile', ha='center', va='top',
             fontsize=10, family='monospace')
    # Second divider line
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

# --- Pixel format selection ---

def select_pixel_format(camera):
    # Cameras can output images in different bit depths (more bits = more shades of grey).
    # We prefer 12-bit, then 10-bit, then 8-bit — whichever the camera supports.
    available = camera.get_pixel_formats()
    for fmt, depth in [(PixelFormat.Mono12, 12), (PixelFormat.Mono10, 10), (PixelFormat.Mono8, 8)]:
        if fmt in available:
            camera.set_pixel_format(fmt)
            return depth  # return how many bits per pixel we ended up with
    sys.exit("Error: no supported monochrome pixel format found on camera")

# --- Main ---

def main():
    # Open a connection to the Allied Vision camera system
    with VmbSystem.get_instance() as vmb:
        cameras = vmb.get_all_cameras()
        if not cameras:
            sys.exit("Error: no Allied Vision cameras found")

        with cameras[0] as camera:  # use the first camera found
            bit_depth = select_pixel_format(camera)
            max_val   = float(2 ** bit_depth - 1)  # maximum possible pixel value (e.g. 4095 for 12-bit)

            # Create the live camera window and connect the mouse click handler
            cv2.namedWindow("GoldenEye")
            btn_coords = (0, 0, 1, 1)  # placeholder; real coords set after first frame
            cv2.setMouseCallback("GoldenEye", mouse_callback, {'btn': btn_coords})

            # --- Main camera loop: runs once per frame until the user presses 'q' ---
            while True:
                global scan_triggered
                frame = camera.get_frame(timeout_ms=2000)   # grab one frame from the camera
                img   = frame.as_numpy_ndarray()             # convert it to a 2-D array of pixel values
                h, w  = img.shape[:2]
                cx, cy = w // 2, h // 2                      # center pixel of the frame

                # The raw image has more than 8 bits, but a screen can only show 0-255.
                # Scale the values down to 8-bit just for display — the raw data is kept for scanning.
                display = (img.astype(float) / max_val * 255).astype(np.uint8)
                display_bgr = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)  # grey -> colour so we can draw green lines

                if scan_triggered:
                    scan_triggered = False

                    # Extract the horizontal row of pixels that runs through the centre of the frame.
                    # This is the line we'll fit the Gaussian to.
                    row   = img[cy].astype(float)

                    # Convert pixel positions to millimetres, centred at zero
                    x_mm  = (np.arange(w) - w / 2) * PIXEL_SIZE_UM * 1e-3

                    fit   = fit_gaussian(x_mm, row)   # attempt to fit a Gaussian to that row
                    show_plot(x_mm, row, fit)          # display the beam profile plot

                # Draw a green crosshair over the live feed to mark the scan line
                cv2.line(display_bgr, (0, cy), (w, cy), (0, 255, 0), 2)   # horizontal line
                cv2.line(display_bgr, (cx, 0), (cx, h), (0, 255, 0), 2)   # vertical line

                # Redraw the SCAN button and update the mouse callback with its current position
                btn_coords = draw_button(display_bgr)
                cv2.setMouseCallback("GoldenEye", mouse_callback, {'btn': btn_coords})

                cv2.imshow("GoldenEye", display_bgr)  # push the frame to the screen

                if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit
                    break

    cv2.destroyAllWindows()  # close the camera window
    plt.close('all')         # close any open plot windows

if __name__ == '__main__':
    main()
