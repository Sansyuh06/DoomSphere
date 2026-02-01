import cv2
import numpy as np


def compute_sgbm_params(baseline_m, focal_px, z_min, z_max, block_sz=5):
    d_max = (baseline_m * focal_px) / z_min
    d_min = (baseline_m * focal_px) / z_max
    
    min_disp = max(0, int(d_min // 16) * 16)
    num_disp = int(np.ceil((d_max - min_disp) / 16) * 16)
    num_disp = max(16, min(256, num_disp))
    
    return {
        'min_disparity': min_disp,
        'num_disparities': num_disp,
        'block_size': block_sz,
        'P1': 8 * 3 * block_sz**2,
        'P2': 32 * 3 * block_sz**2,
        'd_min_theoretical': d_min,
        'd_max_theoretical': d_max
    }


def apply_filters(disp, median_k=5, use_bilateral=False):
    result = disp.copy()
    
    if median_k > 0:
        result = cv2.medianBlur(result.astype(np.float32), median_k)
    
    if use_bilateral:
        d_min, d_max = result.min(), result.max()
        if d_max > d_min:
            norm = ((result - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            norm = cv2.bilateralFilter(norm, 5, 50, 50)
            result = (norm.astype(np.float32) / 255) * (d_max - d_min) + d_min
    
    return result


def make_pointcloud(disparity, Q, z_min, z_max, colors=None, voxel_sz=0.01, outlier_pct=1):
    xyz = cv2.reprojectImageTo3D(disparity, Q)
    z = xyz[:, :, 2]
    mask = (disparity > 0) & np.isfinite(z) & (z > z_min) & (z < z_max)
    
    valid_pts = xyz[mask]
    if len(valid_pts) == 0:
        return np.zeros((0, 3), dtype=np.float32), None, mask
    
    if outlier_pct > 0:
        z_vals = valid_pts[:, 2]
        low = np.percentile(z_vals, outlier_pct)
        high = np.percentile(z_vals, 100 - outlier_pct)
        inliers = (z_vals >= low) & (z_vals <= high)
        valid_pts = valid_pts[inliers]
        valid_colors = colors[mask][inliers] if colors is not None else None
    else:
        valid_colors = colors[mask] if colors is not None else None
    
    if voxel_sz > 0 and len(valid_pts) > 0:
        voxel_idx = (valid_pts / voxel_sz).astype(np.int32)
        _, unique = np.unique(voxel_idx, axis=0, return_index=True)
        valid_pts = valid_pts[unique]
        if valid_colors is not None:
            valid_colors = valid_colors[unique]
    
    return valid_pts, valid_colors, mask


def write_ply(filename, points, colors=None):
    n = len(points)
    if n == 0:
        return
    
    has_color = colors is not None and len(colors) == n
    
    header = f"ply\nformat ascii 1.0\nelement vertex {n}\n"
    header += "property float x\nproperty float y\nproperty float z\n"
    if has_color:
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    header += "end_header\n"
    
    with open(filename, 'w') as f:
        f.write(header)
        for i in range(n):
            x, y, z = points[i]
            if has_color:
                r, g, b = colors[i]
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.4f} {y:.4f} {z:.4f}\n")
