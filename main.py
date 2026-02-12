import cv2
import numpy as np
import time
import os
import config
import calibration as calib
import pointcloud
import rendering
import stereo
import display
import mouse
from camera import ThreadedCamera

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False


def needs_calibration(cfg):
    return not os.path.exists(cfg['calibration']['output_path'])


def run_calibration(cfg):
    print("=" * 50)
    print("  CALIBRATION")
    print("=" * 50)
    
    cam_cfg = cfg['cameras']
    cal_cfg = cfg['calibration']
    
    BOARD = tuple(cal_cfg['chessboard_size'])
    SQUARE = cal_cfg['square_size_meters']
    TARGET = cal_cfg.get('target_captures', 55)
    MIN_CAPS = cal_cfg.get('min_captures', 30)
    
    objp = calib.build_object_points(BOARD, SQUARE)
    objpoints, imgpts_left, imgpts_right = [], [], []

    w, h = cam_cfg['width'], cam_cfg['height']
    cam1 = ThreadedCamera(cam_cfg['left_id'], w, h)
    time.sleep(2.0)
    cam2 = ThreadedCamera(cam_cfg['right_id'], w, h)
    time.sleep(1.0)

    if not cam1.is_ok() or not cam2.is_ok():
        print("ERROR: Could not open cameras!")
        cam1.stop()
        cam2.stop()
        return False
    
    cam1.start()
    cam2.start()
    time.sleep(0.5)

    print(f"Board: {BOARD[0]}x{BOARD[1]}, {SQUARE*1000:.1f}mm")
    print("C=Auto, SPACE=Manual, Q=Done")
    
    count = 0
    auto_mode = False
    last_cap = 0
    frame_n = 0
    found1, found2 = False, False
    corners1, corners2 = None, None
    
    while True:
        try:
            ret1, frame1 = cam1.read()
            ret2, frame2 = cam2.read()
            if not ret1 or frame1 is None or not ret2 or frame2 is None:
                continue
            
            if frame1.shape[:2] != (h, w):
                frame1 = cv2.resize(frame1, (w, h))
            if frame2.shape[:2] != (h, w):
                frame2 = cv2.resize(frame2, (w, h))

            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            frame_n += 1
            if frame_n % 3 == 0:
                found1, corners1 = calib.find_corners(gray1, BOARD)
                found2, corners2 = calib.find_corners(gray2, BOARD)
            
            vis1, vis2 = frame1.copy(), frame2.copy()
            both = found1 and found2
            ok = False
            
            if both:
                ok = calib.check_quality(corners1, w, h) and calib.check_quality(corners2, w, h)
                cv2.drawChessboardCorners(vis1, BOARD, corners1, found1)
                cv2.drawChessboardCorners(vis2, BOARD, corners2, found2)
            
            status = "READY!" if (both and ok) else ("REJECT" if both else "Looking...")
            clr = (0, 255, 0) if (both and ok) else (0, 0, 255)
            cv2.putText(vis1, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, clr, 2)
            cv2.putText(vis1, f"{count}/{TARGET}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if auto_mode:
                cv2.putText(vis1, "AUTO", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            combined = np.hstack((vis1, vis2))
            cv2.imshow('Calibration', combined)
            
            if auto_mode and both and ok and count < TARGET:
                if time.time() - last_cap >= 1.2:
                    c1 = calib.refine_corners(gray1, corners1)
                    c2 = calib.refine_corners(gray2, corners2)
                    objpoints.append(objp)
                    imgpts_left.append(c1)
                    imgpts_right.append(c2)
                    count += 1
                    last_cap = time.time()
                    print(f"[AUTO] {count}/{TARGET}")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                auto_mode = not auto_mode
                last_cap = time.time()
            elif key == 32 and both and ok:
                c1 = calib.refine_corners(gray1, corners1)
                c2 = calib.refine_corners(gray2, corners2)
                objpoints.append(objp)
                imgpts_left.append(c1)
                imgpts_right.append(c2)
                count += 1
                print(f"[MANUAL] {count}/{TARGET}")
        except Exception as e:
            print(f"Error: {e}")

    cam1.stop()
    cam2.stop()
    cv2.destroyAllWindows()
    
    if count < MIN_CAPS:
        print(f"Need at least {MIN_CAPS} captures.")
        return False

    print(f"\nCalibrating with {count} samples...")
    
    ret, K1, D1, K2, D2, R, T, R1, R2, P1, P2, Q = calib.calibrate_stereo(
        objpoints, imgpts_left, imgpts_right, (w, h)
    )
    
    print(f"RMS: {ret:.4f} px, Baseline: {np.linalg.norm(T)*1000:.1f} mm")
    print(f"Quality: {calib.quality_rating(ret)}")
    
    baseline_mm = np.linalg.norm(T) * 1000
    if ret > 1.0:
        print(f"\nCALIBRATION REJECTED! RMS {ret:.2f} is too high (max 1.0)")
        print("The chessboard corners were not detected accurately.")
        print("Use a matte (non-glossy) printed chessboard pattern.")
        return False
    
    if baseline_mm > 500:
        print(f"\nCALIBRATION REJECTED! Baseline {baseline_mm:.0f}mm is unrealistic.")
        print("Expected ~60-120mm. The corner detection was inaccurate.")
        return False
    
    config.save_calibration(cal_cfg['output_path'], K1, D1, K2, D2, R, T, R1, R2, P1, P2, Q, (w, h))
    print("Calibration saved!")
    return True


def run_depth(cfg):
    print("\n" + "=" * 50)
    print("  DEPTH VIEWER")
    print("=" * 50)
    
    cal = config.load_calibration(cfg['calibration']['output_path'])
    if cal is None:
        print("No calibration!")
        return
    
    s_cfg = cfg['stereo']
    d_cfg = cfg['depth']
    c_cfg = cfg['cameras']
    
    K1, D1, K2, D2 = cal['K1'], cal['D1'], cal['K2'], cal['D2']
    R, T = cal['R'], cal['T']
    w, h = cal['image_size']
    
    if 'R1' in cal:
        R1, R2, P1, P2, Q = cal['R1'], cal['R2'], cal['P1'], cal['P2'], cal['Q']
    else:
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T)
    
    map1l, map2l, map1r, map2r = stereo.build_rectify_maps(K1, D1, K2, D2, R1, R2, P1, P2, (w, h))
    
    focal = K1[0, 0]
    baseline = s_cfg.get('baseline_meters', 0.08)
    auto_p = pointcloud.compute_sgbm_params(baseline, focal, d_cfg['z_min'], d_cfg['z_max'])
    
    min_d = s_cfg.get('min_disparity', auto_p['min_disparity'])
    num_d = s_cfg.get('num_disparities', auto_p['num_disparities'])
    blk = s_cfg.get('block_size', 5)
    
    stereo_l = stereo.create_sgbm(min_d, num_d, blk, s_cfg)
    stereo_r, wls = stereo.create_wls(stereo_l, s_cfg) if s_cfg.get('use_wls_filter', True) else (None, None)
    
    print("Opening cameras...")
    cam_l = ThreadedCamera(c_cfg['left_id'], c_cfg['width'], c_cfg['height'])
    time.sleep(2.0)
    cam_r = ThreadedCamera(c_cfg['right_id'], c_cfg['width'], c_cfg['height'])
    time.sleep(1.0)
    
    if not cam_l.is_ok() or not cam_r.is_ok():
        print("ERROR: Cameras failed!")
        cam_l.stop()
        cam_r.stop()
        return
    
    cam_l.start()
    cam_r.start()
    time.sleep(0.5)
    
    pcd, o3d_vis = None, None
    if HAS_O3D:
        o3d_vis = o3d.visualization.Visualizer()
        o3d_vis.create_window("Ghost View", width=800, height=600)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        o3d_vis.add_geometry(pcd)
        opt = o3d_vis.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.1])
        opt.point_size = 2.0
    else:
        cv2.namedWindow("Ghost View")
        cv2.setMouseCallback("Ghost View", mouse.callback)
    
    prev_t = time.time()
    smooth_disp = None
    alpha = 0.3
    ghost_img = np.zeros((600, 600, 3), dtype=np.uint8)
    render_center = None
    smooth_pts = None
    smooth_colors = None
    pts_alpha = 0.4
    
    print("Q=Quit, S=Save")
    
    while True:
        _, frame_l = cam_l.read()
        _, frame_r = cam_r.read()
        if frame_l is None or frame_r is None:
            continue
        
        rect_l = cv2.remap(frame_l, map1l, map2l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(frame_r, map1r, map2r, cv2.INTER_LINEAR)
        
        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
        
        raw_disp = stereo.compute_disparity(stereo_l, stereo_r, wls, gray_l, gray_r)
        
        if s_cfg.get('use_median_filter', True):
            raw_disp = cv2.medianBlur(raw_disp.astype(np.float32), s_cfg.get('median_ksize', 5))
        
        raw_disp = cv2.GaussianBlur(raw_disp, (7, 7), 1.5)
        
        if smooth_disp is None:
            smooth_disp = raw_disp.astype(np.float64)
        else:
            smooth_disp = alpha * raw_disp.astype(np.float64) + (1.0 - alpha) * smooth_disp
        
        disp = smooth_disp.astype(np.float32)
        
        dc = display.colorize_depth(disp, min_d, num_d)
        
        now = time.time()
        fps = 1.0 / (now - prev_t + 1e-5)
        prev_t = now
        display.overlay_fps(dc, fps)
        
        combined = np.hstack((rect_l, dc))
        cv2.imshow("DoomSphere", combined)
        
        xyz = cv2.reprojectImageTo3D(disp, Q)
        z = xyz[:, :, 2]
        valid = (disp > 1.0) & np.isfinite(z)
        
        if np.any(valid):
            if np.median(z[valid]) < 0:
                z = -z
                xyz[:, :, 2] = z
            
            mask = valid & (z > d_cfg['z_min']) & (z < d_cfg['z_max'])
            pts = xyz[mask]
            
            tex = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB)
            pt_colors = tex[mask] / 255.0
            
            if len(pts) > 0:
                max_pts = d_cfg.get('max_points', 200000)
                if len(pts) > max_pts:
                    step = max(1, len(pts) // max_pts)
                    pts = pts[::step]
                    pt_colors = pt_colors[::step]
                
                if smooth_pts is not None and smooth_pts.shape == pts.shape:
                    pts = pts_alpha * pts + (1 - pts_alpha) * smooth_pts
                    pt_colors = pts_alpha * pt_colors + (1 - pts_alpha) * smooth_colors
                smooth_pts = pts.copy()
                smooth_colors = pt_colors.copy()
                
                if HAS_O3D:
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    pcd.colors = o3d.utility.Vector3dVector(pt_colors)
                    o3d_vis.update_geometry(pcd)
                    o3d_vis.poll_events()
                    o3d_vis.update_renderer()
                else:
                    rx, ry = mouse.get_rotation()
                    ghost_img, render_center = rendering.render_cloud(
                        pts, pt_colors, rx=rx, ry=ry, center=render_center
                    )
                    display.overlay_points(ghost_img, len(pts))
        
        cv2.imshow("Ghost View", ghost_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("depth.png", dc)
            print("Saved!")
    
    cam_l.stop()
    cam_r.stop()
    cv2.destroyAllWindows()
    if HAS_O3D:
        o3d_vis.destroy_window()


def main():
    print("=" * 50)
    print("  DOOMSPHERE")
    print("=" * 50)
    
    cfg = config.load_config()
    
    if needs_calibration(cfg):
        print("\nNo calibration found. Starting calibration...\n")
        ok = run_calibration(cfg)
        if not ok:
            print("Calibration failed.")
            return
        print("\nCalibration done! Starting depth...\n")
    else:
        print("\nCalibration found. Starting depth...")
        print("(Delete stereo_params.npz to recalibrate)\n")
    
    run_depth(cfg)


if __name__ == "__main__":
    main()
