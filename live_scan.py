import stereo_utils
import time
import os
import logging
import sys
import argparse
import cv2
import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

logger = stereo_utils.setup_logging("LiveScan")

# =======================================================
# CONFIGURATION LOADED FROM config.json
# =======================================================

def parse_args():
    parser = argparse.ArgumentParser(description="DoomSphere 3D Scanner")
    parser.add_argument("--debug", action="store_true", help="Show extra debug windows")
    return parser.parse_args()

def render_3d_scatter(points, colors, width=640, height=480, rotation_angle=0):
    """
    Fast OpenCV-based 3D scatter plot.
    points: (N, 3) float32
    colors: (N, 3) float32 (0..1)
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if len(points) == 0:
        return canvas

    # 1. Center the cloud
    center = np.mean(points, axis=0)
    pts = points - center
    
    # 2. Rotation (around Y axis)
    theta = np.radians(rotation_angle)
    c, s = np.cos(theta), np.sin(theta)
    
    # Rotation Matrix (Y-axis)
    # x' = x*c + z*s
    # z' = -x*s + z*c
    # y' = y
    x = pts[:, 0] * c + pts[:, 2] * s
    y = pts[:, 1]
    z = -pts[:, 0] * s + pts[:, 2] * c
    
    # 3. Simple Perspective Projection
    # f = focal length (approx pixels)
    f = 400 
    # Move points in front of camera
    z_dist = 2.0 # virtual camera distance
    z_proj = z + z_dist
    
    # Avoid div by zero
    mask = z_proj > 0.1
    x = x[mask]
    y = y[mask]
    z_proj = z_proj[mask]
    cols = colors[mask]
    
    u = (x * f / z_proj) + width // 2
    v = (y * f / z_proj) + height // 2
    
    # 4. Draw
    # Filter out-of-bounds
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid].astype(int)
    v = v[valid].astype(int)
    cols = (cols[valid] * 255).astype(np.uint8)
    
    # Fast drawing: set pixels directly
    canvas[v, u] = cols
    
    return canvas

def main():
    args = parse_args()
    
    # Load Config
    try:
        config = stereo_utils.load_config()
    except Exception:
        return

    PARAM_FILE = config["calibration"]["output_file"]
    CAM_ID_LEFT = config["camera"]["left_id"]
    CAM_ID_RIGHT = config["camera"]["right_id"]
    SGBM_WINDOW_SIZE = config["stereo"]["block_size"]
    MIN_DISPARITY = config["stereo"]["min_disparity"]
    NUM_DISPARITIES = config["stereo"]["num_disparities"]
    VIS_MIN_Z = config["stereo"]["depth_visualizer"]["min_z"]
    VIS_MAX_Z = config["stereo"]["depth_visualizer"]["max_z"]
    width = config["camera"]["width"]
    height = config["camera"]["height"]
    image_size = (width, height)

    if not os.path.exists(PARAM_FILE):
        logger.error(f"{PARAM_FILE} not found. Run calibrate_stereo.py first.")
        return

    # Load calibration data
    logger.info("Loading calibration data...")
    try:
        data = stereo_utils.load_stereo_coefficients(PARAM_FILE)
    except Exception as e:
        logger.error(f"Failed to load calibration data: {e}")
        return
    K1, D1, K2, D2, R, T = data['K1'], data['D1'], data['K2'], data['D2'], data['R'], data['T']
    R1, R2, P1, P2, Q = data['R1'], data['R2'], data['P1'], data['P2'], data['Q']
    image_size = tuple(data['image_size']) # (width, height)
    width, height = image_size

    # Setup Maps for Rectification
    map1_l, map2_l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_16SC2)

    # Setup StereoSGBM
    stereo = cv2.StereoSGBM_create(
        minDisparity=MIN_DISPARITY,
        numDisparities=NUM_DISPARITIES,
        blockSize=SGBM_WINDOW_SIZE,
        P1=8 * 3 * SGBM_WINDOW_SIZE**2,
        P2=32 * 3 * SGBM_WINDOW_SIZE**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY # Added mode for better quality
    )

    # --- PERFORMANCE OPTIMIZATION: Threaded Capture ---
    logger.info("Starting threaded cameras...")
    try:
        cam_l = stereo_utils.ThreadedCamera(CAM_ID_LEFT, width, height).start()
        cam_r = stereo_utils.ThreadedCamera(CAM_ID_RIGHT, width, height).start()
    except Exception as e:
        logger.error(f"Failed to start cameras: {e}")
        return
    
    # Give them a second to warm up
    time.sleep(1.0)

    # Initialize Visualization
    # No more Open3D or Matplotlib objects
    
    use_texture_color = False # Default to "LIDAR" style as requested
    kinect_mode = False # Start in standard mode (Rainbow)
    rot_angle = 0.0 # For 3D view rotation

    # Open3D Visualization Setup - REMOVED
    # if HAS_OPEN3D:
    #     logger.info("Using Open3D for visualization")
    #     vis = o3d.visualization.Visualizer()
    #     vis.create_window(window_name="3D Scanner Point Cloud", width=800, height=600)
    #     pcd = o3d.geometry.PointCloud()
    #     is_geom_added = False
    # else:
    #     logger.warning("Open3D not found (Python 3.14 issue?). Using Matplotlib fallback (Slower).")
    #     plt.ion()
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     ax.set_xlim(-0.5, 0.5)
    #     ax.set_ylim(-0.5, 0.5)
    #     ax.set_zlim(VIS_MIN_Z, VIS_MAX_Z)
    #     scatter_plot = None

    print("\nControls:")
    print("  'q' - Quit")
    print("  'e' - Export .ply")
    print("  's' - Save screenshot of disparity")
    print("  't' - Toggle texture color")
    print("  'k' - Toggle Kinect mode")
    
    # FPS Counter
    prev_time = time.time()
    
    # Setup Trackbars for tuning
    cv2.namedWindow("Disparity")
    cv2.imshow("Disparity", np.zeros((height, width), dtype=np.uint8)) # Force create
    cv2.waitKey(1) # Process event
    
    cv2.createTrackbar("NumDisp", "Disparity", NUM_DISPARITIES, 16*16, lambda x: None)
    cv2.createTrackbar("MinDisp", "Disparity", MIN_DISPARITY + 100, 200, lambda x: None) # Offset 100
    cv2.createTrackbar("Uniqueness", "Disparity", 10, 20, lambda x: None)
    cv2.createTrackbar("Speckle", "Disparity", 100, 200, lambda x: None)
    cv2.createTrackbar("MaxZ (m)", "Disparity", int(VIS_MAX_Z*10), 50, lambda x: None)
    
    use_texture_color = False # Default to "LIDAR" style as requested

    while True:
        # Optimized Read
        ret_l, frame_l = cam_l.read()
        ret_r, frame_r = cam_r.read()

        if not ret_l or not ret_r:
            logger.warning("Waiting for cameras...")
            time.sleep(0.1)
            continue
            
        # Update parameters from trackbars (Handle window closure/race conditions)
        try:
            ndump = cv2.getTrackbarPos("NumDisp", "Disparity")
            mdisp = cv2.getTrackbarPos("MinDisp", "Disparity") - 100
            uniq = cv2.getTrackbarPos("Uniqueness", "Disparity")
            speckle = cv2.getTrackbarPos("Speckle", "Disparity")
            max_z_val = cv2.getTrackbarPos("MaxZ (m)", "Disparity") / 10.0
            if max_z_val < 0.1: max_z_val = 0.5 
            
            # Re-create if closed/missing (catch-22: getTrackbarPos might throw before returning -1)
            if ndump == -1:
                 # Window likely closed by user
                 break
        except cv2.error:
            logger.warning("Disparity window lost. Re-creating...")
            cv2.namedWindow("Disparity")
            cv2.imshow("Disparity", np.zeros((height, width), dtype=np.uint8))
            cv2.createTrackbar("NumDisp", "Disparity", NUM_DISPARITIES, 16*16, lambda x: None)
            cv2.createTrackbar("MinDisp", "Disparity", MIN_DISPARITY + 100, 200, lambda x: None)
            cv2.createTrackbar("Uniqueness", "Disparity", 10, 20, lambda x: None)
            cv2.createTrackbar("Speckle", "Disparity", 100, 200, lambda x: None)
            cv2.createTrackbar("MaxZ (m)", "Disparity", int(VIS_MAX_Z*10), 50, lambda x: None)
            # Use defaults effectively for this frame
            ndump = NUM_DISPARITIES
            mdisp = MIN_DISPARITY
            uniq = 10
            speckle = 100
            max_z_val = VIS_MAX_Z 
        
        # Enforce valid values
        if ndump < 16: ndump = 16
        if ndump % 16 != 0: ndump = (ndump // 16) * 16
        
        stereo.setNumDisparities(ndump)
        stereo.setMinDisparity(mdisp)
        stereo.setUniquenessRatio(uniq)
        stereo.setSpeckleWindowSize(speckle)

        # Force resize to match calibration size (optimization: moved out of lock)
        frame_l = cv2.resize(frame_l, image_size)
        frame_r = cv2.resize(frame_r, image_size)

        # 1. Rectify (Vectorized Remap)
        rect_l = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)

        # 2. Compute Disparity
        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
        
        disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

        # 3. Reproject to 3D (Vectorized Project)
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        
        # 4. Filter bad points (Vectorized Boolean Masking)
        # Using dynamic MaxZ from slider
        mask = (disparity > MIN_DISPARITY) & (points_3d[:,:,2] < max_z_val) & (points_3d[:,:,2] > VIS_MIN_Z) & np.isfinite(points_3d[:,:,2])
        
        points = points_3d[mask]
        
        # Determine Colors ("Lidar Style" vs Texture)
        if use_texture_color:
            colors = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB) 
            colors = colors[mask] / 255.0
        elif kinect_mode:
            # Kinect / Ghost Effect
            # High contrast, "BONE" colormap, auto-ranging for hands
            z_vals = points[:, 2]
            if len(z_vals) > 0:
                # Auto-range to the nearest object (e.g. hand)
                # Find the 10th percentile (closest objects)
                near_z = np.percentile(z_vals, 5) # Focus on closest 5%
                z_range = 0.6 # 60cm interaction zone
                
                z_norm = np.clip((z_vals - near_z) / z_range, 0, 1)
                # Invert logic for BONE: Dark = Background, Light = Foreground?
                # Actually Bone is: Black -> Blue -> White. 
                # We want Near = Bright (White), Far = Dark (Black)
                # So verify normalization
                z_norm_inv = (1.0 - z_norm) * 255
                z_colormap = cv2.applyColorMap(z_norm_inv.astype(np.uint8), cv2.COLORMAP_BONE)
                # Keep in BGR for OpenCV, just fix shape
                colors = z_colormap.reshape(-1, 3) / 255.0
            else:
                 colors = np.zeros((0,3))
        else:
            # Color based on Z-depth (Rainbow/Jet)
            z_vals = points[:, 2]
            if len(z_vals) > 0:
                # Normalize Z to 0-255 range for colormap
                z_norm = cv2.normalize(z_vals, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                # Apply JET colormap
                z_colormap = cv2.applyColorMap(z_norm, cv2.COLORMAP_JET)
                # Keep in BGR for OpenCV
                colors = z_colormap.reshape(-1, 3) / 255.0
            else:
                 colors = np.zeros((0,3))
        
        # 3D Visualization Update - Replaced Open3D/Matplotlib with custom render_3d_scatter
        # Downsample for 3D render specifically to keep FPS high
        render_pts = points
        render_cols = colors
        if len(points) > 50000: # Limit points for faster rendering
             stride = len(points) // 50000
             render_pts = points[::stride]
             render_cols = colors[::stride]
             
        # Auto-rotate
        rot_angle = (rot_angle + 1) % 360
        img_3d = render_3d_scatter(render_pts, render_cols, width=640, height=480, rotation_angle=rot_angle)
        
        cv2.putText(img_3d, "3D VIEW", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        if kinect_mode:
             cv2.putText(img_3d, "KINECT MODE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
        
        cv2.imshow("3D View", img_3d)

        # -------------------------------------------------------
        # 2D Visualization (Disparity Map)
        # -------------------------------------------------------
        if kinect_mode:
             # Show a cool "Ghost" 2D Map
             disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
             disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_BONE)
             cv2.putText(disp_vis, "KINECT MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
             # Standard Rainbow Disparity
             disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
             disp_vis = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
             
        # Center Text
        center_text = "No Object"
        if len(points) > 100:
            centroid = np.mean(points, axis=0)
            center_text = f"X:{centroid[0]:.2f} Y:{centroid[1]:.2f} Z:{centroid[2]:.2f}m"
        
        cv2.putText(disp_vis, center_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if not kinect_mode else (255,255,255), 2)
        cv2.imshow("Disparity", disp_vis)
        
        # -------------------------------------------------------
        # FPS & Input
        # -------------------------------------------------------
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 0.0001) # Added small epsilon to prevent div by zero
        prev_time = curr_time
        
        print(f"FPS: {fps:.1f} | Points: {len(points)}", end='\r')

        # Combined view for rectification check
        combined = np.hstack((rect_l, rect_r))
        for j in range(0, combined.shape[0], 40):
            cv2.line(combined, (0, j), (combined.shape[1], j), (0, 255, 0), 1)
            
        cv2.imshow("Rectified (Alignment Check)", combined)

        # Debug Views
        if args.debug:
            cv2.imshow("Gray Left", gray_l)
            cv2.imshow("Gray Right", gray_r)


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            fn = f"scan_{int(time.time())}.ply"
            stereo_utils.write_ply(fn, points, (colors * 255).astype(np.uint8)) # Use stereo_utils.write_ply
            logger.info(f"Exported {fn}")
        elif key == ord('s'):
            cv2.imwrite("debug_disparity.png", disp_norm)
            logger.info("Saved debug images")
        elif key == ord('t'): # Toggle texture color
            use_texture_color = not use_texture_color
            logger.info(f"Texture Color: {use_texture_color}")
        elif key == ord('k'): # Toggle Kinect mode
            kinect_mode = not kinect_mode
            logger.info(f"Kinect Mode: {kinect_mode}")

    cam_l.stop()
    cam_r.stop()
    # if HAS_OPEN3D: # Removed Open3D cleanup
    #     vis.destroy_window()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
