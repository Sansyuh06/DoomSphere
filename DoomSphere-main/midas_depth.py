"""
midas_depth.py - AI-Based Monocular Depth Estimation
=====================================================
Uses Intel's MiDaS neural network to estimate depth from a SINGLE camera.
No calibration required! Works with any webcam including laptop cameras.

Requirements:
    pip install torch torchvision timm opencv-python

Models available:
    - MiDaS_small: Fast (~15 FPS on CPU, ~60 FPS on GPU)
    - DPT_Hybrid: Balanced
    - DPT_Large: Best quality (~5 FPS on CPU, ~30 FPS on GPU)
"""
import cv2
import numpy as np
import time
import sys
import os

# DEBUG: Print immediately
print("Initializing MiDaS Depth...", flush=True)

# Try to import torch
try:
    print("Importing PyTorch...", end="", flush=True)
    import torch
    import torch.hub
    HAS_TORCH = True
    print(" Done.", flush=True)
except ImportError as e:
    HAS_TORCH = False
    print(f"\nERROR: PyTorch not found: {e}\nInstall: pip install torch torchvision timm", flush=True)
except Exception as e:
    HAS_TORCH = False
    print(f"\nERROR importing PyTorch: {e}", flush=True)

# Mouse state for 3D rotation
mouse_state = {
    'dragging': False, 'last_x': 0, 'last_y': 0,
    'rot_x': -20, 'rot_y': 0
}

def mouse_callback(event, x, y, flags, param):
    global mouse_state
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_state['dragging'] = True
        mouse_state['last_x'], mouse_state['last_y'] = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_state['dragging'] = False
    elif event == cv2.EVENT_MOUSEMOVE and mouse_state['dragging']:
        mouse_state['rot_y'] += (x - mouse_state['last_x']) * 0.5
        mouse_state['rot_x'] += (y - mouse_state['last_y']) * 0.5
        mouse_state['last_x'], mouse_state['last_y'] = x, y

def render_3d_opencv(points, colors, width=600, height=600, rot_x=0, rot_y=0):
    """Render point cloud using OpenCV."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if len(points) == 0:
        return canvas
    
    center = np.median(points, axis=0)
    pts = points - center
    scale = 2.0 / (np.percentile(np.abs(pts), 95) + 0.01)
    pts = pts * scale
    
    # Rotate
    theta_y = np.radians(rot_y)
    cy, sy = np.cos(theta_y), np.sin(theta_y)
    x = pts[:, 0] * cy + pts[:, 2] * sy
    z = -pts[:, 0] * sy + pts[:, 2] * cy
    y = pts[:, 1]
    
    theta_x = np.radians(rot_x)
    cx, sx = np.cos(theta_x), np.sin(theta_x)
    y_new = y * cx - z * sx
    z_new = y * sx + z * cx
    
    # Project
    f = 400
    z_proj = z_new + 3.0
    valid = z_proj > 0.1
    u = (x[valid] * f / z_proj[valid]) + width // 2
    v = (y_new[valid] * f / z_proj[valid]) + height // 2
    c = colors[valid]
    
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, c = u[in_bounds].astype(int), v[in_bounds].astype(int), (c[in_bounds] * 255).astype(np.uint8)
    
    # Draw (Vectorized)
    # Clip indices to be safe
    u = np.clip(u, 0, width - 2)
    v = np.clip(v, 0, height - 2)
    
    # 1. Center pixels
    canvas[v, u] = c
    
    # 2. Right neighbors
    canvas[v, u + 1] = c
    
    # 3. Bottom neighbors
    canvas[v + 1, u] = c
    
    return canvas

def main():
    print("=" * 60, flush=True)
    print("  MiDaS AI DEPTH - Single Camera, No Calibration!", flush=True)
    print("=" * 60, flush=True)
    
    if not HAS_TORCH:
        print("\nERROR: PyTorch is required.")
        print("Install with: pip install torch torchvision timm")
        return
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("  (GPU recommended for real-time performance)")
    
    # Load MiDaS model
    print("\nLoading MiDaS model (first run downloads ~100MB)...", flush=True)
    try:
        model_type = "MiDaS_small"  # Options: MiDaS_small, DPT_Hybrid, DPT_Large
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.to(device)
        midas.eval()
        
        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform
        
        print(f"Loaded: {model_type}")
    except Exception as e:
        print(f"Error loading model: {e}", flush=True)
        import traceback
        traceback.print_exc()
        print("Check your internet connection for first-time download.", flush=True)
        return
    
    # Open camera (try multiple)
    # Open camera (try multiple with backend preference)
    print("\nSearching for working camera...")
    cap = None
    # Prefer DSHOW on Windows for faster access
    backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
    
    available_cameras = [0, 1, 2, 3]
    for cam_id in available_cameras:
        print(f"  Checking Camera {cam_id}...", end="", flush=True)
        temp_cap = cv2.VideoCapture(cam_id, backend)
        
        if temp_cap.isOpened():
            # Try to read a frame to confirm it works
            ret, _ = temp_cap.read()
            if ret:
                print(" SUCCESS!", flush=True)
                cap = temp_cap
                break
            else:
                print(" Opened but failed to read.", flush=True)
                temp_cap.release()
        else:
            print(" Failed to open.", flush=True)
            
    if cap is None:
        # Fallback to standard backend if DSHOW failed
        print("  Retrying with default backend...")
        for cam_id in available_cameras:
            temp_cap = cv2.VideoCapture(cam_id)
            if temp_cap.isOpened() and temp_cap.read()[0]:
                cap = temp_cap
                break
            temp_cap.release()
    
    if cap is None or not cap.isOpened():
        print("ERROR: No camera found!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Setup windows
    cv2.namedWindow("AI Depth View")
    cv2.namedWindow("Ghost View")
    cv2.setMouseCallback("Ghost View", mouse_callback)
    
    print("\n" + "=" * 60)
    print("  Controls:")
    print("    Q - Quit")
    print("    S - Save snapshot")
    print("    Drag on Ghost View to rotate")
    print("=" * 60)
    
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Convert and transform
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img_rgb).to(device)
        
        # Inference
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth = prediction.cpu().numpy()
        
        # Normalize for visualization (invert so closer = brighter)
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-5)
        depth_vis = (depth_norm * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        
        # FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time
        
        # Overlay
        cv2.putText(depth_color, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(depth_color, "MiDaS AI Depth (Single Camera)", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(depth_color, "Near (blue) -> Far (red)", (10, frame.shape[0] - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("AI Depth View", depth_color)
        
        # Generate pseudo-3D point cloud
        h, w = depth.shape
        fx, fy = w, w  # Approximate focal length
        cx, cy = w / 2, h / 2
        
        # Create mesh grid
        u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
        
        # Convert depth to Z (invert because MiDaS gives inverse depth)
        z = 1.0 / (depth + 0.01)
        z = (z - z.min()) / (z.max() - z.min() + 1e-5)  # Normalize to 0-1
        
        # Backproject to 3D
        x = (u_coords - cx) * z / fx
        y = (v_coords - cy) * z / fy
        
        # Sample points (every 4th pixel for speed)
        step = 4
        points = np.stack([
            x[::step, ::step].flatten(),
            y[::step, ::step].flatten(),
            z[::step, ::step].flatten()
        ], axis=1)
        
        # Colors from image
        colors_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[::step, ::step].reshape(-1, 3) / 255.0
        
        # Filter invalid
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.05) & (points[:, 2] < 0.95)
        points = points[valid]
        colors_3d = colors_img[valid]
        
        # Render Ghost View
        if len(points) > 0:
            ghost_img = render_3d_opencv(
                points, colors_3d,
                width=600, height=600,
                rot_x=mouse_state['rot_x'],
                rot_y=mouse_state['rot_y']
            )
            cv2.putText(ghost_img, f"Points: {len(points)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(ghost_img, "Drag to rotate", (10, 580), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.imshow("Ghost View", ghost_img)
        
        # Input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("midas_depth.png", depth_color)
            cv2.imwrite("midas_color.png", frame)
            print("Snapshot saved!")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
