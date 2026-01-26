import cv2
import numpy as np
import time
import stereo_utils
import sys

# Try Open3D
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Open3D not available. Using OpenCV fallback for 3D view.")

# Mouse state for 3D rotation
mouse_state = {
    'dragging': False,
    'last_x': 0,
    'last_y': 0,
    'rot_x': 0,  # Rotation around X (pitch)
    'rot_y': 0   # Rotation around Y (yaw)
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
        mouse_state['rot_y'] += dx * 0.5  # Horizontal drag = Y rotation
        mouse_state['rot_x'] += dy * 0.5  # Vertical drag = X rotation
        mouse_state['last_x'] = x
        mouse_state['last_y'] = y

def render_3d_opencv(points, colors, width=500, height=500, rot_x=0, rot_y=0):
    """Fast OpenCV-based 3D scatter plot with mouse rotation."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if len(points) == 0:
        return canvas
    
    # Center
    center = np.mean(points, axis=0)
    pts = points - center
    
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
    z_proj = z_new + 2.0
    mask = z_proj > 0.1
    
    u = (x[mask] * f / z_proj[mask]) + width // 2
    v = (y_new[mask] * f / z_proj[mask]) + height // 2
    cols = colors[mask]
    
    # Filter bounds
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid].astype(int)
    v = v[valid].astype(int)
    cols = (cols[valid] * 255).astype(np.uint8)
    
    canvas[v, u] = cols
    return canvas

def main():
    print("=== Kinect-Style Depth Camera ===")
    config = stereo_utils.load_config()
    calibration = stereo_utils.load_calibration(config['calibration']['output_path'])
    
    if calibration is None:
        print("Please run calibrate_stereo.py first.")
        return

    # 1. Setup Rectification Maps
    K1, D1, K2, D2 = calibration['K1'], calibration['D1'], calibration['K2'], calibration['D2']
    R, T = calibration['R'], calibration['T']
    width, height = calibration['image_size']
    
    # Compute rectification transforms from saved R, T
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (width, height), R, T)
    
    map1_l, map2_l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_16SC2)

    # 2. Setup SGBM (Left Matcher)
    s_cfg = config['stereo']
    stereo_left = cv2.StereoSGBM_create(
        minDisparity=s_cfg['min_disparity'],
        numDisparities=s_cfg['num_disparities'],
        blockSize=s_cfg['block_size'],
        P1=8 * 3 * s_cfg['block_size']**2,
        P2=32 * 3 * s_cfg['block_size']**2,
        disp12MaxDiff=s_cfg['disp12_max_diff'],
        uniquenessRatio=s_cfg['uniqueness_ratio'],
        speckleWindowSize=s_cfg['speckle_window_size'],
        speckleRange=s_cfg['speckle_range'],
        preFilterCap=s_cfg.get('pre_filter_cap', 63),
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # WLS Filter for smoother disparity (Right matcher needed)
    use_wls = s_cfg.get('use_wls_filter', False)
    wls_filter = None
    stereo_right = None
    if use_wls:
        stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo_left)
        wls_filter.setLambda(s_cfg.get('wls_lambda', 8000))
        wls_filter.setSigmaColor(s_cfg.get('wls_sigma', 1.5))
        print("WLS Filtering ENABLED (Higher quality, slower)")
    else:
        print("WLS Filtering DISABLED")

    # 3. Open Cameras
    c_cfg = config['cameras']
    cam_l = stereo_utils.ThreadedCamera(c_cfg['left_id'], c_cfg['width'], c_cfg['height']).start()
    cam_r = stereo_utils.ThreadedCamera(c_cfg['right_id'], c_cfg['width'], c_cfg['height']).start()
    time.sleep(1.0) # Warmup

    # 4. Open3D Visualizer
    pcd = None
    vis = None
    if HAS_OPEN3D:
        vis = o3d.visualization.Visualizer()
        vis.create_window("Ghost View (3D Point Cloud)", width=800, height=600)
        pcd = o3d.geometry.PointCloud()
        # Initialize
        pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        vis.add_geometry(pcd)
        
        # View Control
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0]) # Black background for ghost look
        opt.point_size = 3.0

    print("Running... Press 'q' to quit, 's' to snapshot, 't' to toggle color.")
    print("Drag mouse on Ghost View to rotate!")
    
    use_texture = False
    d_cfg = config['depth']
    
    # Setup Ghost View window with mouse callback (for OpenCV fallback)
    if not HAS_OPEN3D:
        cv2.namedWindow("Ghost View")
        cv2.setMouseCallback("Ghost View", mouse_callback)
    
    prev_time = time.time()
    
    while True:
        ret_l, frame_l = cam_l.read()
        ret_r, frame_r = cam_r.read()
        
        if not ret_l or not ret_r:
            continue

        # Rectify
        rect_l = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)
        
        # Disparity
        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
        
        # Compute left disparity
        disp_left = stereo_left.compute(gray_l, gray_r)
        
        # Apply WLS filter if enabled
        if use_wls and wls_filter is not None:
            disp_right = stereo_right.compute(gray_r, gray_l)
            disparity = wls_filter.filter(disp_left, gray_l, disparity_map_right=disp_right)
            disparity = disparity.astype(np.float32) / 16.0
        else:
            disparity = disp_left.astype(np.float32) / 16.0
            
        # Median Blur (Remove salt-and-pepper noise)
        if s_cfg.get('use_median_filter', False):
            disparity = cv2.medianBlur(disparity, 5) # Kernel size 5
        
        # 2D Visualization (Kinect Depth Map)
        # Normalize for vis
        disp_vis = (disparity - s_cfg['min_disparity']) / s_cfg['num_disparities']
        disp_vis = np.clip(disp_vis, 0, 1)
        disp_img = (disp_vis * 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_img, cv2.COLORMAP_JET)
        
        # FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time
        cv2.putText(disp_color, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Kinect Depth View", disp_color)
        
        # 3D Point Cloud
        if HAS_OPEN3D:
            # Use robust helper
            valid_xyz, mask = stereo_utils.get_valid_points(disparity, Q, d_cfg['z_min'], d_cfg['z_max'])
            
            if len(valid_xyz) > 0:
                # UNIFORM SUBSAMPLING (Grid-based)
                # Instead of random choice, use stride or voxel grid
                # Numpy stride is fastest
                stride = 3 # Take every 3rd point
                vis_xyz = valid_xyz[::stride]
                
                # Color mapping
                z = vis_xyz[:, 2]
                z_norm = (z - d_cfg['z_min']) / (d_cfg['z_max'] - d_cfg['z_min'])
                z_norm = np.clip(z_norm, 0, 1)
                
                if use_texture:
                    # Need to align colors. get_valid_points returns flattened list.
                    # We need to apply same mask to image.
                    colors_flat = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB).reshape(-1, 3)
                    vis_colors = colors_flat[mask.reshape(-1)][::stride] / 255.0
                else:
                    # Ghost Colors: Blue(Near) -> Cyan(Mid) -> Black(Far)
                    vis_colors = np.zeros((len(z), 3))
                    vis_colors[:, 1] = 1.0 - z_norm**2 # Green fades fast
                    vis_colors[:, 2] = 1.0 - z_norm*0.5 # Blue stays
                
                pcd.points = o3d.utility.Vector3dVector(vis_xyz)
                pcd.colors = o3d.utility.Vector3dVector(vis_colors)
                vis.update_geometry(pcd)
            
            vis.poll_events()
            vis.update_renderer()
        else:
            # OpenCV Fallback 3D View
            valid_xyz, mask = stereo_utils.get_valid_points(disparity, Q, d_cfg['z_min'], d_cfg['z_max'])
            
            if len(valid_xyz) > 0:
                # Voxel/Grid Downsample for OpenCV
                step = 4 # Aggressive downsample for CPU rendering
                render_xyz = valid_xyz[::step]
                
                if len(render_xyz) > 5000: # Cap max points
                     render_xyz = render_xyz[np.linspace(0, len(render_xyz)-1, 5000).astype(int)]
                
                # Ghost Color (Cyan)
                z = render_xyz[:, 2]
                z_norm = (z - d_cfg['z_min']) / (d_cfg['z_max'] - d_cfg['z_min'])
                z_norm = np.clip(z_norm, 0, 1)
                colors = np.zeros((len(z), 3))
                colors[:, 1] = 1.0 - z_norm  
                colors[:, 2] = 1.0 
                
                # Render with mouse-controlled rotation
                ghost_img = render_3d_opencv(
                    render_xyz, colors, 
                    width=500, height=500, 
                    rot_x=mouse_state['rot_x'], 
                    rot_y=mouse_state['rot_y']
                )
                cv2.putText(ghost_img, "GHOST VIEW (Drag to Rotate)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Ghost View", ghost_img)
        else:
            # OpenCV Fallback 3D View
            xyz = cv2.reprojectImageTo3D(disparity, Q)
            mask = (disparity > 0) & (xyz[:,:,2] > d_cfg['z_min']) & (xyz[:,:,2] < d_cfg['z_max']) & np.isfinite(xyz[:,:,2])
            valid_xyz = xyz[mask]
            
            if len(valid_xyz) > 0:
                # Downsample
                target_n = 5000
                if len(valid_xyz) > target_n:
                    choice = np.random.choice(len(valid_xyz), target_n, replace=False)
                    valid_xyz = valid_xyz[choice]
                
                # Ghost Color (Cyan)
                z = valid_xyz[:, 2]
                z_norm = (z - d_cfg['z_min']) / (d_cfg['z_max'] - d_cfg['z_min'])
                z_norm = np.clip(z_norm, 0, 1)
                colors = np.zeros((len(z), 3))
                colors[:, 1] = 1.0 - z_norm  # Green fades
                colors[:, 2] = 1.0  # Blue always
                
                # Render with mouse-controlled rotation
                ghost_img = render_3d_opencv(
                    valid_xyz, colors, 
                    width=500, height=500, 
                    rot_x=mouse_state['rot_x'], 
                    rot_y=mouse_state['rot_y']
                )
                cv2.putText(ghost_img, "GHOST VIEW", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.imshow("Ghost View", ghost_img)

        # Input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            use_texture = not use_texture
        elif key == ord('s'):
            cv2.imwrite("depth.png", disp_color)
            if pcd:
                o3d.io.write_point_cloud("snapshot.ply", pcd)
            print("Snapshot saved.")

    cam_l.stop()
    cam_r.stop()
    cv2.destroyAllWindows()
    if HAS_OPEN3D:
        vis.destroy_window()

if __name__ == "__main__":
    main()
