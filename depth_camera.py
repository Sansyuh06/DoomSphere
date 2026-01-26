"""
depth_camera.py - V2 CHANGES:
- Baseline-aware SGBM parameter calculation
- Disparity filtering (median) before reprojection
- Clean point cloud with voxel downsampling (no more random subsampling)
- Depth legend overlay showing near/far range
- Quality mode toggle (H key for HIGH, F for FAST)
- Better Ghost View with aligned colors and larger points
- Fixed OpenCV fallback 3D renderer
"""
import cv2
import numpy as np
import time
import stereo_utils
from collections import deque

# Try Open3D
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Open3D not available. Using OpenCV fallback for 3D view.")

# ============================================================================
# MOUSE STATE FOR OPENCV 3D VIEW
# ============================================================================

mouse_state = {
    'dragging': False,
    'last_x': 0,
    'last_y': 0,
    'rot_x': -20,  # Start slightly tilted down
    'rot_y': 0
}

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for 3D rotation."""
    global mouse_state
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_state['dragging'] = True
        mouse_state['last_x'] = x
        mouse_state['last_y'] = y
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_state['dragging'] = False
    elif event == cv2.EVENT_MOUSEMOVE and mouse_state['dragging']:
        dx = x - mouse_state['last_x']
        dy = y - mouse_state['last_y']
        mouse_state['rot_y'] += dx * 0.5
        mouse_state['rot_x'] += dy * 0.5
        mouse_state['last_x'] = x
        mouse_state['last_y'] = y

# ============================================================================
# OPENCV 3D RENDERER (FALLBACK)
# ============================================================================

def render_3d_opencv(points, colors, width=600, height=600, rot_x=0, rot_y=0):
    """Render point cloud using OpenCV (fallback when Open3D unavailable)."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    if len(points) == 0:
        return canvas
    
    # Center and scale
    center = np.median(points, axis=0)
    pts = points - center
    scale = 2.0 / (np.percentile(np.abs(pts), 95) + 0.01)
    pts = pts * scale
    
    # Rotate around Y (yaw)
    theta_y = np.radians(rot_y)
    cy, sy = np.cos(theta_y), np.sin(theta_y)
    x = pts[:, 0] * cy + pts[:, 2] * sy
    z = -pts[:, 0] * sy + pts[:, 2] * cy
    y = pts[:, 1]
    
    # Rotate around X (pitch)
    theta_x = np.radians(rot_x)
    cx, sx = np.cos(theta_x), np.sin(theta_x)
    y_new = y * cx - z * sx
    z_new = y * sx + z * cx
    
    # Perspective projection
    f = 400
    z_proj = z_new + 3.0
    valid = z_proj > 0.1
    
    u = (x[valid] * f / z_proj[valid]) + width // 2
    v = (y_new[valid] * f / z_proj[valid]) + height // 2
    c = colors[valid]
    
    # Filter to screen bounds
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[in_bounds].astype(int)
    v = v[in_bounds].astype(int)
    c = (c[in_bounds] * 255).astype(np.uint8)
    
    # Draw points (Vectorized)
    # 1. Center pixels
    canvas[v, u] = c
    
    # 2. Right neighbors (check bounds)
    valid_r = (u + 1) < width
    canvas[v[valid_r], u[valid_r] + 1] = c[valid_r]
    
    # 3. Bottom neighbors (check bounds)
    valid_b = (v + 1) < height
    canvas[v[valid_b] + 1, u[valid_b]] = c[valid_b]
    
    return canvas

# ============================================================================
# DEPTH COLORMAP
# ============================================================================

def depth_to_color(z, z_min, z_max):
    """Convert depth values to ghost-style colors (cyan near, blue far)."""
    z_norm = (z - z_min) / (z_max - z_min + 1e-5)
    z_norm = np.clip(z_norm, 0, 1)
    
    colors = np.zeros((len(z), 3))
    colors[:, 0] = 0.2 * (1 - z_norm)  # R: slight for near
    colors[:, 1] = 0.8 * (1 - z_norm)  # G: bright for near, fades
    colors[:, 2] = 1.0 - 0.3 * z_norm  # B: always high
    
    return colors

# ============================================================================
# TUNING UI
# ============================================================================

def create_tuning_window(s_cfg):
    """Create trackbars for real-time stereo tuning."""
    cv2.namedWindow("Tuning")
    
    def on_trackbar(val): pass
    
    # Core SGBM Params
    cv2.createTrackbar("Num Disp (*16)", "Tuning", s_cfg.get('num_disparities', 128) // 16, 16, on_trackbar)
    cv2.createTrackbar("Block Size", "Tuning", s_cfg.get('block_size', 5), 21, on_trackbar)
    cv2.createTrackbar("Uniqueness", "Tuning", s_cfg.get('uniqueness_ratio', 5), 50, on_trackbar)
    cv2.createTrackbar("Speckle Win", "Tuning", s_cfg.get('speckle_window_size', 200), 500, on_trackbar)
    cv2.createTrackbar("Speckle Rng", "Tuning", s_cfg.get('speckle_range', 2), 20, on_trackbar)
    
    # WLS Params
    cv2.createTrackbar("WLS Lambda", "Tuning", s_cfg.get('wls_lambda', 8000), 20000, on_trackbar)
    cv2.createTrackbar("WLS Sigma", "Tuning", int(s_cfg.get('wls_sigma', 1.2) * 10), 30, on_trackbar)


# ============================================================================
# TERRAIN SCANNER HELPERS
# ============================================================================

def fit_plane_ransac(points, n_iterations=100, distance_threshold=0.01):
    """Fit a plane to points using RANSAC."""
    best_inliers = 0
    best_plane = None
    
    n_points = len(points)
    if n_points < 3: return None

    for _ in range(n_iterations):
        idx = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = points[idx]
        
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6: continue
        normal = normal / norm_len
        d = -np.dot(normal, p1)
        
        distances = np.abs(np.dot(points, normal) + d)
        n_inliers = np.sum(distances < distance_threshold)
        
        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_plane = (*normal, d)
            
    return best_plane

def generate_and_save_terrain(points, res=400):
    """Generate heightmap from points and save files."""
    print("Generating heightmap...")
    x, y = points[:, 0], points[:, 1]
    if len(x) == 0: return
    
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = x_max - x_min + 1e-5
    y_range = y_max - y_min + 1e-5
    
    x_idx = ((x - x_min) / x_range * (res - 1)).astype(int)
    y_idx = ((y_max - y) / y_range * (res - 1)).astype(int)
    
    grid = np.full((res, res), np.nan)
    for i in range(len(points)):
        xi, yi = x_idx[i], y_idx[i]
        if 0 <= xi < res and 0 <= yi < res:
            if np.isnan(grid[yi, xi]) or points[i, 2] > grid[yi, xi]:
                grid[yi, xi] = points[i, 2]
                
    # Fill holes
    grid_filled = grid.copy()
    for _ in range(3):
        mask_nan = np.isnan(grid_filled)
        kernel = np.ones((3, 3)) / 9
        grid_smooth = cv2.filter2D(np.nan_to_num(grid_filled), -1, kernel)
        grid_filled[mask_nan] = grid_smooth[mask_nan]
        
    # Visualize
    g_min, g_max = np.nanmin(grid_filled), np.nanmax(grid_filled)
    grid_norm = (grid_filled - g_min) / (g_max - g_min + 1e-5)
    grid_img = (np.nan_to_num(grid_norm) * 255).astype(np.uint8)
    heightmap = cv2.applyColorMap(grid_img, cv2.COLORMAP_JET)
    heightmap[np.isnan(grid_filled)] = [30, 30, 30]
    
    # Save
    cv2.imwrite("heightmap.png", heightmap)
    print(f"Saved heightmap.png (Range: {g_min*1000:.1f}mm - {g_max*1000:.1f}mm)")


def render_top_down(points, width=400, height=400, x_range=(-1.0, 1.0), z_range=(0.0, 3.0)):
    """
    Render a fast Top-Down (Map) view of the point cloud.
    X-axis = Horizontal (Camera Left/Right)
    Z-axis = Vertical (Camera Depth)
    Color = Y-axis (Height)
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if len(points) == 0: return canvas

    # Filter to view range
    mask = (points[:, 0] > x_range[0]) & (points[:, 0] < x_range[1]) & \
           (points[:, 2] > z_range[0]) & (points[:, 2] < z_range[1])
    pts = points[mask]
    if len(pts) == 0: return canvas

    # Map X, Z to U, V
    # U: -1 to 1 -> 0 to width
    u = ((pts[:, 0] - x_range[0]) / (x_range[1] - x_range[0]) * width).astype(int)
    # V: 0 to 3 -> height to 0 (Depth up)
    v = ((1.0 - (pts[:, 2] - z_range[0]) / (z_range[1] - z_range[0])) * height).astype(int)

    # Color by Height (Y is down, so -Y is up)
    h = -pts[:, 1]
    h_min, h_max = -0.5, 0.5 # Range: floor to 50cm up
    h_norm = np.clip((h - h_min) / (h_max - h_min), 0, 1)
    
    # Simple heatmap color (Blue=Low, Red=High)
    colors = np.zeros((len(h), 3), dtype=np.uint8)
    colors[:, 0] = (255 * (1 - h_norm)).astype(np.uint8) # B
    colors[:, 2] = (255 * h_norm).astype(np.uint8)       # R
    
    # Draw (Vectorized scatter)
    # Clip indices
    u = np.clip(u, 0, width-1)
    v = np.clip(v, 0, height-1)
    
    # Draw
    canvas[v, u] = colors
    
    # Draw Grid Lines
    cv2.line(canvas, (0, height//2), (width, height//2), (50, 50, 50), 1) # 1.5m line
    cv2.line(canvas, (width//2, 0), (width//2, height), (50, 50, 50), 1) # Center line
    
    # Labels
    cv2.putText(canvas, "Top View (Map)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, f"Range: {x_range[0]}m to {x_range[1]}m width", (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    return canvas


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 50)
    print("  KINECT-STYLE DEPTH CAMERA V2")
    print("=" * 50)
    
    config = stereo_utils.load_config()
    calibration = stereo_utils.load_calibration(config['calibration']['output_path'])
    
    if calibration is None:
        print("Please run calibrate_stereo.py first.")
        return
    
    # Load settings
    s_cfg = config['stereo']
    d_cfg = config['depth']
    c_cfg = config['cameras']
    quality_mode = config.get('quality_mode', 'high')
    
    # Get calibration matrices
    K1, D1, K2, D2 = calibration['K1'], calibration['D1'], calibration['K2'], calibration['D2']
    R, T = calibration['R'], calibration['T']
    width, height = calibration['image_size']
    
    # Compute rectification (use saved if available, else compute)
    if 'R1' in calibration:
        R1, R2, P1, P2, Q = calibration['R1'], calibration['R2'], calibration['P1'], calibration['P2'], calibration['Q']
    else:
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (width, height), R, T)
    
    # Build rectification maps
    map1_l, map2_l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_16SC2)
    
    # Compute baseline-aware SGBM params
    focal_px = K1[0, 0]
    baseline = s_cfg.get('baseline_meters', 0.08)
    auto_params = stereo_utils.compute_sgbm_params(baseline, focal_px, d_cfg['z_min'], d_cfg['z_max'])
    print(f"Auto SGBM: minDisp={auto_params['min_disparity']}, numDisp={auto_params['num_disparities']}")
    print(f"  (Theoretical disparity range: {auto_params['d_min_theoretical']:.1f} - {auto_params['d_max_theoretical']:.1f})")
    
    # Use config values but fall back to auto if not specified
    min_disp = s_cfg.get('min_disparity', auto_params['min_disparity'])
    num_disp = s_cfg.get('num_disparities', auto_params['num_disparities'])
    
    # Create SGBM matchers
    stereo_left = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=s_cfg.get('block_size', 5),
        P1=8 * 3 * s_cfg.get('block_size', 5)**2,
        P2=32 * 3 * s_cfg.get('block_size', 5)**2,
        disp12MaxDiff=s_cfg.get('disp12_max_diff', 5),
        uniquenessRatio=s_cfg.get('uniqueness_ratio', 5),
        speckleWindowSize=s_cfg.get('speckle_window_size', 200),
        speckleRange=s_cfg.get('speckle_range', 2),
        preFilterCap=s_cfg.get('pre_filter_cap', 31),
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # WLS filter setup
    use_wls = s_cfg.get('use_wls_filter', True)
    wls_filter = None
    stereo_right = None
    if use_wls:
        try:
            stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo_left)
            wls_filter.setLambda(s_cfg.get('wls_lambda', 8000))
            wls_filter.setSigmaColor(s_cfg.get('wls_sigma', 1.2))
            print("WLS Filtering: ENABLED")
        except AttributeError:
            print("WLS Filtering: DISABLED (opencv-contrib not installed)")
            use_wls = False
    
    # Helper to update matcher from trackbars
    def update_matcher_params():
        if not cv2.getWindowProperty("Tuning", cv2.WND_PROP_VISIBLE):
            return
            
        nd = max(1, cv2.getTrackbarPos("Num Disp (*16)", "Tuning")) * 16
        bs = cv2.getTrackbarPos("Block Size", "Tuning")
        if bs % 2 == 0: bs += 1 # Ensure odd
        if bs < 3: bs = 3
        
        stereo_left.setNumDisparities(nd)
        stereo_left.setBlockSize(bs)
        stereo_left.setP1(8 * 3 * bs**2)
        stereo_left.setP2(32 * 3 * bs**2)
        stereo_left.setUniquenessRatio(cv2.getTrackbarPos("Uniqueness", "Tuning"))
        stereo_left.setSpeckleWindowSize(cv2.getTrackbarPos("Speckle Win", "Tuning"))
        stereo_left.setSpeckleRange(cv2.getTrackbarPos("Speckle Rng", "Tuning"))
        
        if wls_filter:
            wls_filter.setLambda(cv2.getTrackbarPos("WLS Lambda", "Tuning"))
            sigma = cv2.getTrackbarPos("WLS Sigma", "Tuning") / 10.0
            wls_filter.setSigmaColor(sigma)
            
        # Also update right matcher for WLS consistency
        if stereo_right:
             stereo_right.setP1(8 * 3 * bs**2)
             stereo_right.setP2(32 * 3 * bs**2)
             stereo_right.setBlockSize(bs)
             stereo_right.setNumDisparities(nd)
             stereo_right.setSpeckleWindowSize(cv2.getTrackbarPos("Speckle Win", "Tuning"))
             stereo_right.setSpeckleRange(cv2.getTrackbarPos("Speckle Rng", "Tuning"))
    
    # Open cameras
    print(f"\nOpening cameras {c_cfg['left_id']} and {c_cfg['right_id']}...")
    cam_l = stereo_utils.ThreadedCamera(c_cfg['left_id'], c_cfg['width'], c_cfg['height']).start()
    cam_r = stereo_utils.ThreadedCamera(c_cfg['right_id'], c_cfg['width'], c_cfg['height']).start()
    time.sleep(1.0)
    
    # Open3D visualizer
    pcd = None
    vis = None
    if HAS_OPEN3D:
        vis = o3d.visualization.Visualizer()
        vis.create_window("Ghost View", width=800, height=600)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.05, 0.05, 0.1])
        opt.point_size = 2.5
    else:
        cv2.namedWindow("Ghost View")
        cv2.setMouseCallback("Ghost View", mouse_callback)
    
    # State
    use_texture = False
    show_invalid = False
    prev_time = time.time()
    
    # Temporal averaging buffer
    disparity_buffer = deque(maxlen=5)
    use_temporal = True
    use_confidence = True  # Texture-based confidence
    confidence_thresh = 15.0
    
    print("\n" + "=" * 50)
    print("  Controls:")
    print("    Q - Quit")
    print("    T - Toggle texture/depth color")
    print("    S - Save snapshot")
    print("    H - HIGH quality mode")
    print("    F - FAST mode")
    print("    I - Toggle invalid pixel display")
    print("    A - Toggle temporal averaging (Default: ON)")
    print("    C - Toggle confidence masking (Default: ON)")
    print("    U - Toggle Tuning Mode UI")
    print("    M - Start Terrain Scan (30 frames)")
    print("=" * 50)
    
    # Tuning state
    tuning_active = False 
    
    # Scanning state
    scanning = False
    scan_frames = 0
    scan_limit = 30
    scan_sum = None
    scan_count = None 

    while True:
        ret_l, frame_l = cam_l.read()
        ret_r, frame_r = cam_r.read()
        
        if frame_l is None or frame_r is None:
            continue
        
        # Rectify
        rect_l = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)
        
        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture confidence (gradient magnitude)
        if use_confidence:
            sobel_x = cv2.Sobel(gray_l, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_l, cv2.CV_64F, 0, 1, ksize=3)
            texture_mag = np.sqrt(sobel_x**2 + sobel_y**2)
            low_texture_mask = texture_mag < confidence_thresh
        
        if tuning_active:
            try:
                update_matcher_params()
            except: pass

        # Compute disparity
        disp_left = stereo_left.compute(gray_l, gray_r)
        
        if use_wls and wls_filter is not None:
            disp_right = stereo_right.compute(gray_r, gray_l)
            disparity = wls_filter.filter(disp_left, gray_l, disparity_map_right=disp_right)
            disparity = disparity.astype(np.float32) / 16.0
        else:
            disparity = disp_left.astype(np.float32) / 16.0
        
        # Temporal Averaging
        if use_temporal:
            disparity_buffer.append(disparity)
            if len(disparity_buffer) > 1:
                # Fast averaging using sum logic
                # Convert to accumulation accumulator then divide
                disparity = np.mean(disparity_buffer, axis=0)
        
        if s_cfg.get('use_median_filter', True) or s_cfg.get('use_bilateral_filter', False):
            disparity = stereo_utils.apply_disparity_filters(
                disparity, 
                median_ksize=s_cfg.get('median_ksize', 5) if s_cfg.get('use_median_filter', True) else 0,
                use_bilateral=s_cfg.get('use_bilateral_filter', False)
            )
            
        # Apply confidence mask (set low texture to invalid)
        if use_confidence:
            disparity[low_texture_mask] = 0
        
        # Depth visualization
        disp_vis = (disparity - min_disp) / num_disp
        disp_vis = np.clip(disp_vis, 0, 1)
        disp_img = (disp_vis * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(disp_img, cv2.COLORMAP_JET)
        
        # Show invalid as black if enabled
        if show_invalid:
            invalid_mask = disparity <= 0
            depth_color[invalid_mask] = [0, 0, 0]
        
        # FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time
        
        # Overlay text
        cv2.putText(depth_color, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(depth_color, f"Near: {d_cfg['z_min']}m (blue) | Far: {d_cfg['z_max']}m (red)", 
                    (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(depth_color, f"Mode: {quality_mode.upper()}", (width - 120, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("Kinect Depth View", depth_color)
        
        # Generate point cloud
        colors_for_cloud = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB) if use_texture else None
        
        # Reproject to 3D
        xyz = cv2.reprojectImageTo3D(disparity, Q)
        z = xyz[:, :, 2]
        valid_disp = disparity > 0
        valid_z = np.isfinite(z) & valid_disp
        
        if np.any(valid_z):
            z_valid = z[valid_z]
            
            # Auto-detect if Z is negative (inverted camera setup)
            # Use absolute Z for filtering if most values are negative
            if np.median(z_valid) < 0:
                z = -z  # Flip sign
                xyz[:, :, 2] = z
            
            # Now filter with positive Z range
            z_min_actual = d_cfg['z_min']
            z_max_actual = d_cfg['z_max']
            mask = valid_z & (z > z_min_actual) & (z < z_max_actual)
            points = xyz[mask]
            point_colors = colors_for_cloud[mask] if colors_for_cloud is not None else None
            
            # Debug every few seconds
            if int(time.time()) % 5 == 0:
                print(f"[DEBUG] Points: {len(points)}, Z: {z[valid_z].min():.2f} to {z[valid_z].max():.2f} m")
        else:
            points = np.zeros((0, 3))
            point_colors = None
        
        if len(points) > 0:
            # Color based on mode
            if use_texture and point_colors is not None:
                vis_colors = point_colors / 255.0
            else:
                vis_colors = depth_to_color(points[:, 2], d_cfg['z_min'], d_cfg['z_max'])
            
            # Cap points for performance
            max_pts = d_cfg.get('max_points', 50000)
            if len(points) > max_pts:
                idx = np.linspace(0, len(points) - 1, max_pts).astype(int)
                points = points[idx]
                vis_colors = vis_colors[idx]
            
            if HAS_OPEN3D:
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(vis_colors)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
            else:
                ghost_img = render_3d_opencv(
                    points, vis_colors,
                    width=600, height=600,
                    rot_x=mouse_state['rot_x'],
                    rot_y=mouse_state['rot_y']
                )
                cv2.putText(ghost_img, f"Points: {len(points)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(ghost_img, "Drag to rotate", (10, 580), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                cv2.imshow("Ghost View", ghost_img)
        else:
            # Show empty ghost view with debug info
            if not HAS_OPEN3D:
                ghost_img = np.zeros((600, 600, 3), dtype=np.uint8)
                cv2.putText(ghost_img, "No valid points", (200, 280), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(ghost_img, f"Disp range: {disparity.min():.1f} - {disparity.max():.1f}", (150, 320), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                cv2.putText(ghost_img, f"Z range: {d_cfg['z_min']} - {d_cfg['z_max']}m", (180, 360), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                cv2.imshow("Ghost View", ghost_img)
        
        # Terrain Scanning Logic
        if scanning:
            # Accumulate
            valid = disparity > 0
            if scan_sum is None:
                scan_sum = np.zeros_like(disparity)
                scan_count = np.zeros_like(disparity)
            
            scan_sum[valid] += disparity[valid]
            scan_count[valid] += 1
            scan_frames += 1
            
            cv2.putText(depth_color, f"SCANNING: {scan_frames}/{scan_limit}", (width//2 - 100, height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.imshow("Kinect Depth View", depth_color) # Update overlay
            
            if scan_frames >= scan_limit:
                print("Processing scan... Please wait.")
                scanning = False
                scan_count[scan_count == 0] = 1
                avg_disp = scan_sum / scan_count
                
                # Generate cloud
                s_pts, s_colors, _ = stereo_utils.make_clean_pointcloud(
                    avg_disp, Q, d_cfg['z_min'], d_cfg['z_max'], outlier_percentile=2
                )
                
                if len(s_pts) > 100:
                    # Ground plane logic
                    z_thresh = np.percentile(s_pts[:, 2], 20)
                    g_cands = s_pts[s_pts[:, 2] < z_thresh]
                    if len(g_cands) > 100:
                        plane = fit_plane_ransac(g_cands)
                        if plane:
                            a, b, c, d = plane
                            s_pts[:, 2] = s_pts[:, 0]*a + s_pts[:, 1]*b + s_pts[:, 2]*c + d
                    
                    # Save outputs
                    stereo_utils.write_ply("terrain_cloud.ply", s_pts, s_colors)
                    generate_and_save_terrain(s_pts)
                    print("Scan saved to terrain_cloud.ply and heightmap.png")
                else:
                    print("Scan failed: Not enough points.")
                    
                scan_sum = None
                scan_count = None

        # Live Heightmap Window (3rd View)
        if len(points) > 0:
            map_img = render_top_down(points, x_range=(-1.0, 1.0), z_range=(0.0, 3.0))
            cv2.imshow("Live Heightmap", map_img)
            
        cv2.imshow("Ghost View", ghost_img)
        
        # Handle input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            use_texture = not use_texture
            print(f"Texture mode: {'ON' if use_texture else 'OFF'}")
        elif key == ord('i'):
            show_invalid = not show_invalid
            print(f"Show invalid: {'ON' if show_invalid else 'OFF'}")
        elif key == ord('a'):
            use_temporal = not use_temporal
            disparity_buffer.clear()
            print(f"Temporal Averaging: {'ON' if use_temporal else 'OFF'}")
        elif key == ord('c'):
            use_confidence = not use_confidence
            print(f"Confidence Masking: {'ON' if use_confidence else 'OFF'}")
        elif key == ord('u'):
            tuning_active = not tuning_active
            if tuning_active:
                create_tuning_window(s_cfg)
            else:
                cv2.destroyWindow("Tuning")
            print(f"Tuning Mode: {'ON' if tuning_active else 'OFF'}")
        elif key == ord('m'):
            if not scanning:
                print("Starting Terrain Scan (30 frames)...")
                scanning = True
                scan_frames = 0
                scan_sum = None
        elif key == ord('s'):
            cv2.imwrite("depth_snapshot.png", depth_color)
            if len(points) > 0:
                stereo_utils.write_ply("snapshot.ply", points, 
                                       (vis_colors * 255).astype(np.uint8) if vis_colors is not None else None)
            print("Snapshot saved!")
        elif key == ord('h'):
            quality_mode = 'high'
            print("Quality mode: HIGH")
        elif key == ord('f'):
            quality_mode = 'fast'
            print("Quality mode: FAST")
    
    # Cleanup
    cam_l.stop()
    cam_r.stop()
    cv2.destroyAllWindows()
    if HAS_OPEN3D:
        vis.destroy_window()

if __name__ == "__main__":
    main()
