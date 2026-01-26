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
        speckleRange=32
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

    # Open3D Visualization Setup
    if HAS_OPEN3D:
        logger.info("Using Open3D for visualization")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="3D Scanner Point Cloud", width=800, height=600)
        pcd = o3d.geometry.PointCloud()
        is_geom_added = False
    else:
        logger.warning("Open3D not found (Python 3.14 issue?). Using Matplotlib fallback (Slower).")
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(VIS_MIN_Z, VIS_MAX_Z)
        scatter_plot = None

    print("\nControls:")
    print("  'q' - Quit")
    print("  'e' - Export .ply")
    print("  's' - Save screenshot of disparity")
    
    # FPS Counter
    prev_time = time.time()
    
    while True:
        # Optimized Read
        ret_l, frame_l = cam_l.read()
        ret_r, frame_r = cam_r.read()

        if not ret_l or not ret_r:
            logger.warning("Waiting for cameras...")
            time.sleep(0.1)
            continue

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
        mask = (disparity > MIN_DISPARITY) & (points_3d[:,:,2] < VIS_MAX_Z) & (points_3d[:,:,2] > VIS_MIN_Z)
        
        points = points_3d[mask]
        colors = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB) 
        colors = colors[mask] / 255.0

        if len(points) > 0:
            if HAS_OPEN3D:
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                
                if not is_geom_added:
                    vis.add_geometry(pcd)
                    is_geom_added = True
                
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
            else:
                # Matplotlib Fallback (Downsample heavily for speed)
                step = 10 
                pts = points[::step]
                cols = colors[::step]
                
                if scatter_plot:
                    scatter_plot.remove()
                
                if len(pts) > 0:
                    scatter_plot = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=cols, s=1)
                
                plt.draw()
                plt.pause(0.001)

        # Update FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Visualization in 2D
        disp_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.putText(disp_norm, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Disparity", disp_norm)
        
        # Combined view for rectification check
        combined = np.hstack((rect_l, rect_r))
        cv2.line(combined, (0, 100), (2*width, 100), (0, 255, 0), 1)
        cv2.imshow("Rectified Left/Right", combined)

        # Debug Views
        if args.debug:
            cv2.imshow("Gray Left", gray_l)
            cv2.imshow("Gray Right", gray_r)


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            fn = f"scan_{int(time.time())}.ply"
            o3d.io.write_point_cloud(fn, pcd)
            logger.info(f"Exported {fn}")
        elif key == ord('s'):
            cv2.imwrite("debug_disparity.png", disp_norm)
            logger.info("Saved debug images")

    cam_l.stop()
    cam_r.stop()
    if HAS_OPEN3D:
        vis.destroy_window()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
