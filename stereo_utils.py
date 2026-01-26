import cv2
import numpy as np
import json
import threading
import time
import os

def load_config(path="config.json"):
    """Loads configuration from a JSON file."""
    if not os.path.exists(path):
        print(f"Config file {path} not found.")
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def save_calibration(path, K1, D1, K2, D2, R, T, R1, R2, P1, P2, Q, image_size):
    """Saves full calibration and rectification matrices to an NPZ file."""
    np.savez(path, 
             K1=K1, D1=D1, K2=K2, D2=D2, 
             R=R, T=T, 
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, 
             image_size=image_size)
    print(f"Calibration saved to {path} with full rectification data.")

def load_calibration(path):
    """Loads calibration matrices from an NPZ file."""
    if not os.path.exists(path):
        print(f"Calibration file {path} not found.")
        return None
    data = np.load(path)
    return {
        'K1': data['K1'], 'D1': data['D1'],
        'K2': data['K2'], 'D2': data['D2'],
        'R': data['R'], 'T': data['T'], 
        'R1': data['R1'], 'R2': data['R2'], 
        'P1': data['P1'], 'P2': data['P2'],
        'Q': data['Q'], 
        'image_size': tuple(data['image_size'])
    }

class ThreadedCamera:
    """Efficiently grabs frames in a separate thread."""
    def __init__(self, src, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) # Optimized for Windows
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # Try MJPG for speed
        self.ret, self.frame = self.cap.read()
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            time.sleep(0.001)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap.isOpened():
            self.cap.release()

def get_valid_points(disparity, Q, z_min=0.3, z_max=2.0, roi=None):
    """
    Reprojects disparity to 3D, filters outliers, and returns robust XYZ points.
    args:
        disparity: float32 disparity map
        Q: 4x4 reprojection matrix
        z_min, z_max: depth clipping range
        roi: (optional) tuple (x, y, w, h) from stereoRectify to crop invalid borders
    returns:
        points: (N, 3) float32
        mask: boolean mask of original shape
    """
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    
    # Extract channels
    x, y, z = points_3d[:,:,0], points_3d[:,:,1], points_3d[:,:,2]
    
    # 1. Mask invalid disparity
    mask = (disparity > 0)
    
    # 2. Mask invalid Z
    mask &= (z > z_min) & (z < z_max) & np.isfinite(z)
    
    # 3. Mask ROI (remove black borders from rectification)
    if roi is not None:
        rx, ry, rw, rh = roi
        h, w = disparity.shape
        # Create grid to check ROI
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        mask &= (grid_x >= rx) & (grid_x < rx+rw) & (grid_y >= ry) & (grid_y < ry+rh)
    
    valid_points = points_3d[mask]
    return valid_points, mask

def write_ply(filename, points, colors=None):
    """Simple PLY writer for point clouds."""
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
""".format(len(points))
    
    if colors is not None:
        header += """property uchar red
property uchar green
property uchar blue
"""
    header += """end_header
"""
    
    with open(filename, 'w') as f:
        f.write(header)
        for i in range(len(points)):
            x, y, z = points[i]
            if colors is not None:
                r, g, b = colors[i]
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.4f} {y:.4f} {z:.4f}\n")
    print(f"Saved {filename}")
