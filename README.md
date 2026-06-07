# DoomSphere

## Problem Statement

Modern depth sensing is expensive and often locked behind proprietary hardware. This project aims to turn a pair of low-cost USB cameras into a functioning stereo depth system. The key challenge is to calibrate two cameras precisely, compute reliable disparity maps, and render a stable 3D view in real time.

## Solution

DoomSphere combines stereo camera calibration, rectification, depth estimation, and point cloud rendering into one Python application.

- `main.py` is the heart of the system: it runs a calibration workflow as needed, then switches to real-time depth processing.
- `calibration.py` detects chessboard patterns, refines corner positions, and filters poor sample frames.
- `stereo.py` builds a StereoSGBM matcher, rectifies images, and computes disparity maps.
- `pointcloud.py` converts disparity into a 3D point cloud and optionally saves results as PLY.
- `display.py` colorizes depth, overlays status text and FPS, and supports a fallback renderer when Open3D is unavailable.

The result is an end-to-end stereo depth pipeline that can run on consumer hardware and work with standard camera pairs.

## Tech Stack

- Python 3.x
- OpenCV (`opencv-python`) for camera capture, calibration, image rectification, disparity computation, and visualization
- NumPy for numerical arrays, camera matrix handling, and point cloud math
- Open3D (`open3d`) as an optional 3D viewer for higher-quality point cloud rendering

## How It Works

1. **Calibration phase**
   - The system uses a 7x7 chessboard pattern with a configured square size.
   - It captures synchronized stereo pairs and verifies corner quality before accepting samples.
   - After collecting enough good frames, it computes intrinsic and extrinsic stereo calibration parameters.
   - A stereo rectification matrix and reprojection matrix (`Q`) are saved to `stereo_params.npz`.

2. **Depth generation**
   - Video from both cameras is undistorted and rectified using the saved calibration.
   - The app computes a disparity map with OpenCV StereoSGBM and optional WLS filtering.
   - Depth values are derived by reprojecting pixels through the stereo `Q` matrix.
   - The display shows the left camera image side-by-side with a false-color depth map.

3. **3D visualization**
   - If Open3D is installed, a live 3D point cloud window is displayed.
   - Otherwise, a custom `Ghost View` renderer shows a pseudo-3D projection of the point cloud.
   - The app supports saving the current depth map as `depth.png`.

## How to Run

1. Create and activate a Python environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Configure your cameras in `config.json`:
   - `left_id` and `right_id` should match your connected camera indices.
   - `width`, `height`, and `fps` control capture resolution and framerate.
   - Adjust `baseline_meters` and `z_min`/`z_max` for your physical stereo rig.

4. Start the project:

```bash
python launcher.py
```

5. Choose `Start (Calibrate + Depth)` to run the full workflow.
   - If no calibration exists, the app automatically begins camera calibration.
   - After calibration, it proceeds into real-time stereo depth viewing.
   - Press `Q` to quit and `S` to save the current depth image.

## Build and Development Notes

- `launcher.py` is a simple menu wrapper around `main.py` for ease of use.
- Calibration quality is validated by chessboard size, corner location, and target RMS error.
- Disparity smoothing and median filtering reduce noise for a cleaner 3D output.
- The pipeline is intentionally modular so calibration, depth, rendering, and point cloud logic remain separate and easy to extend.

## Impact

This project demonstrates how to create an accessible stereo depth viewer without dedicated hardware like LiDAR or Intel RealSense. It is useful for:

- rapid prototyping of computer vision and robotics systems
- educational demos showing stereo geometry and 3D reconstruction
- low-cost 3D scanning and environment awareness for hobby projects
- testing depth-based interaction or augmented reality concepts with standard USB cameras

By providing a calibration-first workflow and an optional Open3D visualization path, DoomSphere bridges the gap between raw stereo camera input and practical depth sensing.
