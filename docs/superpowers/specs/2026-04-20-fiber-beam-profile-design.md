# Fiber Beam Profile — Design Spec
*2026-04-20*

## Context

Testing.py is a webcam-based prototype to validate the concept of scanning a fiber lens and plotting its brightness profile. The end goal is to measure beam profiles from a fiber end face using a monochrome GoldenEye camera with a 50× scope. This spec covers the webcam prototype phase.

The current code reads raw brightness along the horizontal center row and draws a primitive cv2 chart. We need to:
1. Fit a Gaussian curve to that data
2. Compute the 1/e² beam diameter
3. Display a clean scientific plot (matplotlib) matching the reference plot style

## Architecture

Single file: `Testing.py`

Three logical sections:

### 1. Camera (cv2) — unchanged
- `draw_button(frame)` — SCAN button top-right
- `mouse_callback(...)` — sets `scan_triggered = True`
- `pixel_to_brightness(bgr_pixel)` — luminance from BGR
- Main loop: capture → draw → scan on click → display

### 2. Gaussian Fitting (new — scipy)
Function: `fit_gaussian(x_vals, brightness_vals) → dict`

Model: `f(x) = A · exp(−2·(x−x₀)²/w²) + B`

- `A` = amplitude above baseline
- `x₀` = center (near 0 since x is centered)
- `w` = 1/e² beam radius in pixels
- `B` = background baseline

Returns: `{success, A, x0, w, B}` or `{success: False}` on fit failure.

Initial parameter guess: peak value for A, pixel at max brightness for x₀, frame_width/8 for w, min brightness for B.

### 3. Plot Display (new — matplotlib, non-blocking)
Function: `show_plot(x_vals, brightness_vals, fit_result)`

- X-axis: pixel position centered at 0 (`index − width/2`)
- Y-axis: brightness 0–255
- Blue line: raw brightness data
- Red curve: fitted Gaussian (if fit succeeded)
- Red dashed horizontal: 1/e² level at `A/e² + B ≈ 0.135·A + B`
- Text annotation: `1/e² Beam Diameter = {2w:.1f} px`
- Title: `Line Scan — Beam Profile`
- If fit failed: raw data only + "Fit failed" text label

Uses `plt.ion()` + `plt.show(block=False)` so the cv2 camera window stays live.

## GoldenEye Migration Path

When switching to the GoldenEye monochrome camera:
- Change `cv2.VideoCapture(0)` to the GoldenEye source/SDK
- Monochrome input: skip BGR→luminance conversion; use raw pixel value directly
- If 10/12-bit depth: update Y-axis range accordingly (0–1023 or 0–4095)
- Add pixel calibration constant (µm/pixel from scope magnification) to convert X-axis from pixels to physical units

## Critical Files

- `Testing.py` — only file to modify

## Dependencies

- `cv2` (already used)
- `numpy` (already used)
- `scipy.optimize.curve_fit` (new — `pip install scipy`)
- `matplotlib.pyplot` (new — `pip install matplotlib`)

## Verification

1. Run `python Testing.py`
2. Point webcam at a bright light source or fiber end
3. Click SCAN
4. Confirm matplotlib window appears with:
   - Raw brightness line (blue)
   - Fitted Gaussian (red)
   - Dashed 1/e² horizontal line
   - Beam diameter annotation
5. Try SCAN on a flat/dark scene — confirm "Fit failed" appears gracefully
6. Close with `q` — confirm both windows close cleanly
