import cv2
import numpy as np
import time
import sys
import stereo_utils
from stereo_utils import ThreadedCamera
import logging
import matplotlib.pyplot as plt

# Try Open3D for visualization, but don't crash if missing
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Terrain Scanner (LiDAR Style)")
    parser.add_argument("--frames", type=int, default=50, help="Frames to accumulate")
    parser.add_argument("--max-z", type=float, default=1.5, help="Clipping Z distance")
    parser.add_argument("--voxel-size", type=float, default=0.005, help="Voxel Grid Size")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = stereo_utils.setup_logging("TerrainScan")
    logger.info("Initializing Terrain Scanner...")

    # Load Config
    config = stereo_utils.load_config()
    terrain_cfg = config.get("terrain", {})
    stereo_cfg = config.get("stereo", {})
    
    # Priority: CLI > Config > Default
    FRAMES_TO_ACCUMULATE = args.frames if args.frames != 50 else terrain_cfg.get("frames_to_accumulate", 50)
    VOXEL_SIZE = args.voxel_size if args.voxel_size != 0.005 else terrain_cfg.get("voxel_size", 0.005)
    MAX_Z = args.max_z if args.max_z != 1.5 else terrain_cfg.get("max_z", 1.5)
    
    # Actually just use CLI args heavily if provided, simplistic override
    FRAMES_TO_ACCUMULATE = args.frames
    VOXEL_SIZE = args.voxel_size
    MAX_Z = args.max_z
    
    GROUND_PERCENTILE = terrain_cfg.get("ground_percentile", 2)

    # Load Calibration
    if not os.path.exists("stereo_params.npz"):
        logger.error("Calibration not found! Run calibrate_stereo.py first.")
        return
        
    calib = stereo_utils.load_stereo_coefficients("stereo_params.npz")
    K1, D1, K2, D2 = calib['K1'], calib['D1'], calib['K2'], calib['D2']
    R1, P1, R2, P2, Q = calib['R1'], calib['P1'], calib['R2'], calib['P2'], calib['Q']
    image_size = tuple(calib['image_size'])

    # Setup Stereo Matcher
    min_disp = stereo_cfg.get("min_disparity", 0)
    num_disp = stereo_cfg.get("num_disparities", 128)
    block_size = stereo_cfg.get("block_size", 11)
    
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Initialize Cameras
    cam1 = ThreadedCamera(config["camera"]["left_id"], 640, 480).start()
    cam2 = ThreadedCamera(config["camera"]["right_id"], 640, 480).start()
    
    # Undistortion Maps
    leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

    all_points = []
    
    logger.info(f"Starting capture of {FRAMES_TO_ACCUMULATE} frames...")
    logger.info("KEEP THE OBJECT STILL!")
    time.sleep(1) # Give camera time to settle

    try:
        for i in range(FRAMES_TO_ACCUMULATE):
            ret1, frame1 = cam1.read()
            ret2, frame2 = cam2.read()
            
            if not ret1 or not ret2:
                time.sleep(0.01)
                continue

            # Rectify
            imgL = cv2.remap(frame1, leftMapX, leftMapY, cv2.INTER_LINEAR)
            imgR = cv2.remap(frame2, rightMapX, rightMapY, cv2.INTER_LINEAR)
            
            # Disparity
            gray_l = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
            
            # Compute disparity (using simple SGBM here for speed, or match depth_camera)
            # Better to use the SAME matcher config as depth_camera for consistency
            disp = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
            
            # Use Robust Filtering
            pts, mask = stereo_utils.get_valid_points(disp, Q, z_min=0.2, z_max=MAX_Z)
            
            # Simple downsample
            if len(pts) > 20000:
                pts = pts[::int(len(pts)/20000)]
            
            if len(pts) > 0:
                all_points.append(pts)
            
            print(f"Captured frame {i+1}/{FRAMES_TO_ACCUMULATE} - {len(pts)} pts", end='\r')
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        logger.warning("\nCapture interrupted.")
    finally:
        cam1.stop()
        cam2.stop()
        
    print() # Newline
    if not all_points:
        logger.error("No points captured.")
        return

    # Consolidate
    logger.info("Consolidating point cloud...")
    cloud = np.vstack(all_points)
    original_count = len(cloud)
    logger.info(f"Total raw points: {original_count}")

    # Voxel Downsampling (NumPy style)
    if len(cloud) > 0 and VOXEL_SIZE > 0:
        logger.info(f"Voxel downsampling ({VOXEL_SIZE}m)...")
        # Snap to grid
        voxel_indices = (cloud / VOXEL_SIZE).astype(int)
        # Find unique voxels
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        cloud = cloud[unique_indices]
        logger.info(f"Downsampled to {len(cloud)} points")
    
    if len(cloud) == 0:
        logger.error("No valid points found after filtering/downsampling.")
        return

    # Filter NaNs explicity (just in case)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]

    # Ground Normalization
    logger.info("Normalizing height...")
    if len(cloud) == 0:
        logger.error("Cloud empty after removing NaNs.")
        return

    # Assume the lowest X% of points are the table/ground
    z_coords = cloud[:, 2]
    ground_level = np.nanpercentile(z_coords, GROUND_PERCENTILE)
    cloud[:, 2] -= ground_level # Shift ground to 0
    
    # Clip filtering
    # Remove things below the table (noise) and too high
    high_mask = (cloud[:, 2] > -0.05) & (cloud[:, 2] < MAX_Z - ground_level)
    cloud = cloud[high_mask]
    
    # Statistical Outlier Removal (NumPy approximation if Open3D missing)
    # Simple radius check is O(N^2) in numpy, expensive.
    # We will rely on the accumulation to average out noise, and voxel grid to clean up.
    
    # Colors
    logger.info("Generating Terrain Colors...")
    if len(cloud) > 0:
        p_max = np.nanpercentile(cloud[:, 2], 98)
        colors = stereo_utils.apply_colormap_to_depth(
            cloud[:, 2], 
            scale_min=0.0, 
            scale_max=p_max, # Ignore top 2% outliers for color scale
            cmap_name='jet'
        )
    else:
        colors = np.zeros(cloud.shape, dtype=np.uint8)

    # Save PLY
    stereo_utils.write_ply("terrain_cloud.ply", cloud, colors)
    
    # Generate Heightmap Image
    logger.info("Generating Heightmap Image...")
    generate_heightmap(cloud, config)
    
    logger.success("Done! Saved 'terrain_cloud.ply' and 'heightmap.png'")

    # Visualize
    visualize(cloud, colors)

def generate_heightmap(points, config):
    # Project to XY grid
    if len(points) == 0: return
    
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Define bounds (centered usually)
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    
    # Resolution
    res = 1000 # 1000px image
    
    # Mapping
    x_img = ((x - min_x) / (max_x - min_x) * (res - 1)).astype(int)
    # flip Y for image coords
    y_img = ((max_y - y) / (max_y - min_y) * (res - 1)).astype(int) 
    
    # Grid initialization
    grid = np.zeros((res, res), dtype=np.float32) - 999
    
    # Simple max-z projection (not efficient but works)
    # For speed, we can assume points are somewhat ordered or just loop
    # Faster: use lexicographical sort and keep last (highest Z) for each X,Y?
    # Actually, we can just iterate. N is small enough (~50k)
    
    # However, to be fast in numpy:
    # We want max Z for each unique (x_img, y_img).
    # Combine x,y into header
    pixel_indices = y_img * res + x_img
    
    # Pass 1: find max z for each linear index
    # We can use pandas groupby, or simple loop
    # Or strict updates
    output_buffer = np.full(res*res, -np.inf)
    
    # Use reduceat? No. 
    # Just scatter max. `np.maximum.at`
    np.maximum.at(output_buffer, pixel_indices, z)
    
    # Reshape
    grid = output_buffer.reshape((res, res))
    
    # Mask invalid
    mask = grid == -np.inf
    grid[mask] = np.nan
    
    # Normalize for image
    valid_z = grid[~mask]
    if len(valid_z) == 0: return
    
    z_min, z_max = np.nanmin(valid_z), np.nanmax(valid_z)
    norm_grid = (grid - z_min) / (z_max - z_min)
    norm_grid = np.nan_to_num(norm_grid, 0)
    
    # Colorize
    bgr = stereo_utils.apply_colormap_to_depth(norm_grid, 0, 1, 'jet')
    # Reshape back to image
    img = bgr.reshape((res, res, 3))
    
    # Set background to black (or white)
    img[mask.reshape((res, res))] = [0, 0, 0]
    
    cv2.imwrite("heightmap.png", img)


def visualize(points, colors):
    if HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        
        # Set view looking down
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Terrain Result")
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        vis.run()
        vis.destroy_window()
    else:
        # Matplotlib Fallback
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Downsample strictly for plot
        idx = np.random.choice(len(points), min(len(points), 5000), replace=False)
        pts = points[idx]
        col = colors[idx] / 255.0
        
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=col, s=1)
        ax.set_title("Terrain Scan (Matplotlib)")
        plt.show()

import os
if __name__ == "__main__":
    main()
