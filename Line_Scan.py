# --- IMPORTS ---
# vmbpy is Allied Vision's Python SDK for controlling professional cameras like the GoldenEye.
# VmbSystem is the entry point — you always need it to interact with any Allied Vision camera.
# PixelFormat lets you request a specific image format (e.g. greyscale, 8-bit, 12-bit).
from vmbpy import VmbSystem, PixelFormat
import cv2       # OpenCV — used later for displaying or processing camera frames
import numpy as np  # numpy — used for fast math on image data arrays

# --- SECTION: Initialise the Vimba Camera System ---

# Get the single global Vimba system object (there is always exactly one).
# Think of this as "turning on" the camera management system.
vmb = VmbSystem.get_instance()

# Manually open the Vimba system using Python's context manager protocol.
# Normally you'd write `with VmbSystem.get_instance() as vmb:` but here we
# open it manually so the connection stays alive for the rest of the script.
vmb.__enter__()

# --- SECTION: Find Connected Cameras ---

# Ask Vimba to list all cameras currently plugged in
cameras = vmb.get_all_cameras()

if not cameras:
    print("Error: No cameras found")
else:
    print(f"Found {len(cameras)} camera(s)")

    # Connect to the first camera in the list (index 0).
    # If you have multiple cameras you could change 0 to 1, 2, etc.
    camera = cameras[0]

    # Open the camera connection (same manual context manager approach as above)
    camera.__enter__()

    print(f"Connected to camera: {camera.get_name()}")

    # --- TODO: Add frame capture and beam scan code here ---
    # This is where you will later call camera.get_frame() to grab images
    # and feed them into the Gaussian fitting pipeline from Testing.py.
