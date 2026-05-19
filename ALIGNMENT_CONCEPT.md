# Alignment Concept: Physical Distance Line on Webcam Feed

## End Goal
Draw a vertical reference line at exactly **0.75 mm** from the center crosshair on a live camera feed.  
Purpose: visually align a **V-groove array tip** and the edge of the **gripper tip** with a precise 0.75 mm gap between them — without needing access to the real system yet.

---

## The Core Problem: Pixels ≠ Millimeters

A camera sees the world in **pixels**, not millimeters. To draw a line at a physical distance, you must know how many pixels equal 1 mm at your working distance. This is called **`mm_per_pixel`** (or its inverse, `pixels_per_mm`).

```
offset_pixels = round(0.75 / mm_per_pixel)
line_x = cx + offset_pixels
```

---

## How the Production Code Already Does This

In `Line_Scan.py` (lines 21–26), the system uses a known microscope camera with fixed optics:

```python
PIXEL_SIZE  = 5.0    # Alvium G1-030 sensor pixel pitch (µm)
OBJECTIVE_MAG   = 100    # Mitutoyo 100x objective
ZOOM_MAG        = 5.25   # Optem Fusion at max zoom
CAMERA_TUBE_MAG = 1.0

UM_PER_PIXEL = PIXEL_SIZE / (OBJECTIVE_MAG * ZOOM_MAG * CAMERA_TUBE_MAG)
# Result: ~0.0095 µm per pixel
```

This works because all optical parameters are known and fixed.

---

## Why a Webcam Is Different

A standard webcam (`cv2.VideoCapture(0)`) has:
- Unknown sensor pixel pitch
- No fixed magnification or objective
- Field of view that changes with distance

So `mm_per_pixel` is **not a constant** — it depends entirely on how far the camera sits from the objects.

---

## The Fix: One-Time Calibration

Place a known reference object (ruler, object of known width) in front of the camera **at your exact working distance**. Then:

1. Count how many pixels it spans → `span_pixels`
2. Know its real size → `span_mm`
3. Calculate: `mm_per_pixel = span_mm / span_pixels`

After calibration, the line position is accurate **as long as the camera doesn't move**.

---

## Implementation Sketch (for the real system)

```python
# --- Calibration constant (set once per camera/distance setup) ---
MM_PER_PIXEL = 0.01  # example: measure and replace this

# --- Inside the frame loop ---
h, w = frame.shape[:2]
cx, cy = w // 2, h // 2

# Green center crosshair
cv2.line(frame, (cx, 0), (cx, h), (0, 255, 0), 1)

# Red reference line at 0.75 mm to the right
offset_px = round(0.75 / MM_PER_PIXEL)
cv2.line(frame, (cx + offset_px, 0), (cx + offset_px, h), (0, 0, 255), 1)
```

---

## Key Takeaways

| | Webcam (Testing.py) | Production (Line_Scan.py) |
|---|---|---|
| Pixel size known? | No — needs calibration | Yes — from hardware specs |
| mm_per_pixel | Empirical (measure a reference) | Calculated from optics |
| Works for alignment? | Yes, if camera is fixed | Yes |

- The concept is **valid and not overcomplicated**
- Webcam prototype is a legitimate way to test the idea
- The only requirement is a **one-time calibration** at a fixed working distance
- Once `mm_per_pixel` is known, the 0.75 mm line is just `cx + round(0.75 / mm_per_pixel)`
