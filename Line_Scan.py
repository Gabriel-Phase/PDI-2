from vmbpy import VmbSystem, PixelFormat
import cv2
import numpy as np

# Get the global Vimba system instance and open it
vmb = VmbSystem.get_instance()
vmb.__enter__()

# List all connected cameras
cameras = vmb.get_all_cameras()

if not cameras:
    print("Error: No cameras found")
else:
    print(f"Found {len(cameras)} camera(s)")

    # Connect to the first available camera
    camera = cameras[0]
    camera.__enter__()

    print(f"Connected to camera: {camera.get_name()}")
