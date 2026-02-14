"""
stereo_utils.py - V2 CHANGES:
- Added compute_sgbm_params() for baseline-aware parameter calculation
- Added apply_disparity_filters() for median/bilateral filtering
- Added make_clean_pointcloud() for robust 3D conversion with voxel downsampling
- ThreadedCamera now tries multiple backends (DSHOW, MSMF, ANY)
- save/load_calibration now includes R1/R2/P1/P2 for direct use
"""
import cv2
import numpy as np
import json
import threading
import time
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config(path="config.json"):
    """Loads configuration from a JSON file."""
    if not os.path.exists(path):
        print(f"Config file {path} not found.")
        return {}
    with open(path, 'r') as f:
        return json.load(f)

# ============================================================================
# CALIBRATION I/O
# ============================================================================

def save_calibration(path, K1, D1, K2, D2, R, T, R1, R2, P1, P2, Q, image_size):
    """Saves full calibration and rectification matrices to an NPZ file."""
    np.savez(path, 
             K1=K1, D1=D1, K2=K2, D2=D2, 
             R=R, T=T, 
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, 
             image_size=image_size)
    print(f"Calibration saved to {path}")

def load_calibration(path):
    """Loads calibration matrices from an NPZ file."""
    if not os.path.exists(path):
        print(f"Calibration file {path} not found.")
        return None
    data = np.load(path)
    
    # Handle both old and new format
    result = {
        'K1': data['K1'], 'D1': data['D1'],
        'K2': data['K2'], 'D2': data['D2'],
        'R': data['R'], 'T': data['T'], 
        'Q': data['Q'], 
        'image_size': tuple(data['image_size'])
    }
    
    # New format includes rectification matrices
    if 'R1' in data:
        result['R1'] = data['R1']
        result['R2'] = data['R2']
        result['P1'] = data['P1']
        result['P2'] = data['P2']
    
    return result

# ============================================================================
# SGBM PARAMETER CALCULATION
# ============================================================================

def compute_sgbm_params(baseline_m, focal_px, z_min, z_max, block_size=5):
    """
    Compute optimal SGBM parameters based on physical setup.
    
    Args:
        baseline_m: Camera separation in meters (e.g., 0.08)
        focal_px: Focal length in pixels (from K matrix, typically K[0,0])
        z_min: Minimum depth in meters
        z_max: Maximum depth in meters
        block_size: SGBM block size (odd number, 3-11)
    
    Returns:
        dict with min_disparity, num_disparities, and other SGBM params
    
    Formula: disparity = baseline * focal / depth
    """
    # Maximum disparity (at z_min - closest objects)
    d_max = (baseline_m * focal_px) / z_min
    # Minimum disparity (at z_max - far objects)
    d_min = (baseline_m * focal_px) / z_max
    
    # Round to multiples of 16
    min_disparity = max(0, int(d_min // 16) * 16)
    num_disparities = int(np.ceil((d_max - min_disparity) / 16) * 16)
    num_disparities = max(16, min(256, num_disparities))  # Clamp to valid range
    
    return {
        'min_disparity': min_disparity,
        'num_disparities': num_disparities,
        'block_size': block_size,
        'P1': 8 * 3 * block_size**2,
        'P2': 32 * 3 * block_size**2,
        'd_min_theoretical': d_min,
        'd_max_theoretical': d_max
    }

# ============================================================================
# DISPARITY FILTERING
# ============================================================================

def apply_disparity_filters(disparity, median_ksize=5, use_bilateral=False):
    """
    Apply noise-reduction filters to disparity map.
    
    Args:
        disparity: Float32 disparity map
        median_ksize: Kernel size for median filter (odd number, 0 to disable)
        use_bilateral: Whether to apply bilateral filter for edge-preserving smoothing
    
    Returns:
        Filtered disparity map
    """
    result = disparity.copy()
    
    # Median filter removes salt-and-pepper noise
    if median_ksize > 0:
        # cv2.medianBlur requires uint8 or float32
        result = cv2.medianBlur(result.astype(np.float32), median_ksize)
    
    # Bilateral filter preserves edges while smoothing
    if use_bilateral:
        # Normalize to 0-255 range for bilateral
        d_min, d_max = result.min(), result.max()
        if d_max > d_min:
            norm = ((result - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            norm = cv2.bilateralFilter(norm, 5, 50, 50)
            result = (norm.astype(np.float32) / 255) * (d_max - d_min) + d_min
    
    return result

# ============================================================================
# POINT CLOUD GENERATION
# ============================================================================

def make_clean_pointcloud(disparity, Q, z_min, z_max, colors=None, 
                          voxel_size=0.01, outlier_percentile=1):
    """
    Generate a clean, filtered point cloud from disparity.
    
    Args:
        disparity: Float32 disparity map
        Q: 4x4 reprojection matrix from stereoRectify
        z_min, z_max: Depth clipping range in meters
        colors: Optional (H, W, 3) uint8 BGR image for texture
        voxel_size: Voxel grid size for downsampling (0 to disable)
        outlier_percentile: Remove points outside this percentile range
    
    Returns:
        points: (N, 3) float32 array
        point_colors: (N, 3) uint8 array or None
        mask: (H, W) boolean mask of valid pixels
    """
    # Reproject to 3D
    xyz = cv2.reprojectImageTo3D(disparity, Q)
    
    # Build validity mask
    z = xyz[:, :, 2]
    mask = (disparity > 0) & np.isfinite(z) & (z > z_min) & (z < z_max)
    
    # Extract valid points
    valid_xyz = xyz[mask]
    
    if len(valid_xyz) == 0:
        return np.zeros((0, 3), dtype=np.float32), None, mask
    
    # Remove outliers by percentile
    if outlier_percentile > 0:
        z_valid = valid_xyz[:, 2]
        p_low = np.percentile(z_valid, outlier_percentile)
        p_high = np.percentile(z_valid, 100 - outlier_percentile)
        inlier_mask = (z_valid >= p_low) & (z_valid <= p_high)
        valid_xyz = valid_xyz[inlier_mask]
        
        # Also filter colors if provided
        if colors is not None:
            valid_colors = colors[mask][inlier_mask]
        else:
            valid_colors = None
    else:
        valid_colors = colors[mask] if colors is not None else None
    
    # Voxel grid downsampling
    if voxel_size > 0 and len(valid_xyz) > 0:
        # Quantize to voxel grid
        voxel_indices = (valid_xyz / voxel_size).astype(np.int32)
        _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
        valid_xyz = valid_xyz[unique_idx]
        if valid_colors is not None:
            valid_colors = valid_colors[unique_idx]
    
    return valid_xyz, valid_colors, mask

# ============================================================================
# THREADED CAMERA
# ============================================================================

class ThreadedCamera:
    """Efficiently grabs frames in a separate thread with fallback backends."""
    
    def __init__(self, src, width=640, height=480):
        self.src = src
        self.cap = None
        self.ret = False
        self.frame = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Try multiple backends
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for backend in backends:
            self.cap = cv2.VideoCapture(src, backend)
            if self.cap.isOpened():
                print(f"Camera {src} opened with backend {backend}")
                break
        
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            self.ret, self.frame = self.cap.read()
        else:
            print(f"Warning: Could not open camera {src}")
            self.cap = None

    def start(self):
        if self.cap is None:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            time.sleep(0.001)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()

# ============================================================================
# PLY WRITER
# ============================================================================

def write_ply(filename, points, colors=None):
    """Simple PLY writer for point clouds."""
    n = len(points)
    if n == 0:
        print(f"Warning: No points to save to {filename}")
        return
    
    has_colors = colors is not None and len(colors) == n
    
    header = f"""ply
format ascii 1.0
element vertex {n}
property float x
property float y
property float z
"""
    if has_colors:
        header += """property uchar red
property uchar green
property uchar blue
"""
    header += "end_header\n"
    
    with open(filename, 'w') as f:
        f.write(header)
        for i in range(n):
            x, y, z = points[i]
            if has_colors:
                r, g, b = colors[i]
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.4f} {y:.4f} {z:.4f}\n")
    
    print(f"Saved {filename} ({n} points)")

# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

def get_valid_points(disparity, Q, z_min=0.3, z_max=2.0, roi=None):
    """Legacy function - use make_clean_pointcloud instead."""
    points, _, mask = make_clean_pointcloud(disparity, Q, z_min, z_max, 
                                            voxel_size=0, outlier_percentile=0)
    return points, mask
