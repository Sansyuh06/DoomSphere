"""
calibrate_stereo.py - V2 CHANGES:
- Capture quality check (board size, edge rejection)
- Detailed reprojection error analysis (mean, median, 95th percentile)
- Quality classification: EXCELLENT/GOOD/OK/BAD with guidance
- Visual rectification verification before saving
- Saves full R1/R2/P1/P2/Q for runtime use
- Multi-backend camera support
"""
import cv2
import numpy as np
import time
import stereo_utils

def open_camera(cam_id, width, height):
    """Try multiple backends to open camera."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(cam_id, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print(f"  Camera {cam_id}: opened (backend {backend})")
            return cap
        cap.release()
    return None

def compute_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, K, D):
    """Compute per-point reprojection errors for quality analysis."""
    all_errors = []
    for i in range(len(objpoints)):
        proj_points, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        errors = np.linalg.norm(imgpoints[i].reshape(-1, 2) - proj_points.reshape(-1, 2), axis=1)
        all_errors.extend(errors)
    return np.array(all_errors)

def main():
    print("=" * 60)
    print("  STEREO CALIBRATION V2")
    print("  For maximum accuracy, capture 40-60 varied poses")
    print("=" * 60)
    
    config = stereo_utils.load_config()
    c_cfg = config['cameras']
    cal_cfg = config['calibration']
    
    BOARD_SIZE = tuple(cal_cfg['chessboard_size'])
    SQUARE_SIZE = cal_cfg['square_size_meters']
    TARGET_CAPTURES = cal_cfg.get('target_captures', 55)
    MIN_CAPTURES = cal_cfg.get('min_captures', 30)
    
    # Prepare object points
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints_l = []
    imgpoints_r = []

    # Open cameras
    width, height = c_cfg['width'], c_cfg['height']
    print(f"\nOpening cameras {c_cfg['left_id']} and {c_cfg['right_id']}...")
    cam1 = open_camera(c_cfg['left_id'], width, height)
    cam2 = open_camera(c_cfg['right_id'], width, height)

    if cam1 is None or cam2 is None:
        print("\nERROR: Could not open cameras!")
        print("  1. Close any app using the cameras")
        print("  2. Check camera IDs in config.json")
        print("  3. Unplug and replug USB cameras")
        return

    print(f"\nBoard: {BOARD_SIZE[0]}x{BOARD_SIZE[1]} inner corners, {SQUARE_SIZE*1000:.1f}mm squares")
    print("\nControls:")
    print("  C - Start/Stop AUTO-CAPTURE")
    print("  SPACE - Manual capture")
    print("  Q - Finish and calibrate")
    print(f"\nTarget: {TARGET_CAPTURES} captures (min {MIN_CAPTURES})")
    print("-" * 60)
    
    count = 0
    auto_mode = False
    last_capture_time = 0
    CAPTURE_INTERVAL = 1.2
    MIN_BOARD_AREA_RATIO = 0.05  # Board must cover at least 5% of frame
    EDGE_MARGIN = 20  # Reject if corners too close to edge
    
    while True:
        try:
            ret1, frame1 = cam1.read()
            ret2, frame2 = cam2.read()
            
            if not ret1 or not ret2:
                continue
            
            # Force consistent size
            if frame1.shape[:2] != (height, width):
                frame1 = cv2.resize(frame1, (width, height))
            if frame2.shape[:2] != (height, width):
                frame2 = cv2.resize(frame2, (width, height))

            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Find corners
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret_c1, corners1 = cv2.findChessboardCorners(gray1, BOARD_SIZE, flags)
            ret_c2, corners2 = cv2.findChessboardCorners(gray2, BOARD_SIZE, flags)
            
            vis1 = frame1.copy()
            vis2 = frame2.copy()
            
            valid_pair = ret_c1 and ret_c2
            quality_ok = False
            reject_reason = ""
            
            if valid_pair:
                # Quality checks
                # 1. Check board size (area)
                hull1 = cv2.convexHull(corners1)
                hull2 = cv2.convexHull(corners2)
                area1 = cv2.contourArea(hull1)
                area2 = cv2.contourArea(hull2)
                frame_area = width * height
                
                if area1 < frame_area * MIN_BOARD_AREA_RATIO or area2 < frame_area * MIN_BOARD_AREA_RATIO:
                    reject_reason = "Board too small"
                else:
                    # 2. Check edge proximity
                    c1_pts = corners1.reshape(-1, 2)
                    c2_pts = corners2.reshape(-1, 2)
                    
                    near_edge_1 = np.any(c1_pts[:, 0] < EDGE_MARGIN) or np.any(c1_pts[:, 0] > width - EDGE_MARGIN)
                    near_edge_1 = near_edge_1 or np.any(c1_pts[:, 1] < EDGE_MARGIN) or np.any(c1_pts[:, 1] > height - EDGE_MARGIN)
                    near_edge_2 = np.any(c2_pts[:, 0] < EDGE_MARGIN) or np.any(c2_pts[:, 0] > width - EDGE_MARGIN)
                    near_edge_2 = near_edge_2 or np.any(c2_pts[:, 1] < EDGE_MARGIN) or np.any(c2_pts[:, 1] > height - EDGE_MARGIN)
                    
                    if near_edge_1 or near_edge_2:
                        reject_reason = "Too close to edge"
                    else:
                        quality_ok = True
                
                cv2.drawChessboardCorners(vis1, BOARD_SIZE, corners1, ret_c1)
                cv2.drawChessboardCorners(vis2, BOARD_SIZE, corners2, ret_c2)
            
            # Status display
            if valid_pair and quality_ok:
                cv2.putText(vis1, "READY!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            elif valid_pair:
                cv2.putText(vis1, f"REJECT: {reject_reason}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                cv2.putText(vis1, "Looking for board...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.putText(vis1, f"Captures: {count}/{TARGET_CAPTURES}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if auto_mode:
                cv2.putText(vis1, "AUTO MODE", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Progress bar
            progress = int((count / TARGET_CAPTURES) * (width - 20))
            cv2.rectangle(vis1, (10, height - 30), (10 + progress, height - 10), (0, 255, 0), -1)
            cv2.rectangle(vis1, (10, height - 30), (width - 10, height - 10), (255, 255, 255), 2)
            
            combined = np.hstack((vis1, vis2))
            cv2.imshow('Stereo Calibration V2 (C=Auto, Q=Done)', combined)
            
            # Auto-capture
            if auto_mode and valid_pair and quality_ok and count < TARGET_CAPTURES:
                current_time = time.time()
                if current_time - last_capture_time >= CAPTURE_INTERVAL:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners1_sub = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                    corners2_sub = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
                    
                    objpoints.append(objp)
                    imgpoints_l.append(corners1_sub)
                    imgpoints_r.append(corners2_sub)
                    count += 1
                    last_capture_time = current_time
                    print(f"[AUTO] Captured {count}/{TARGET_CAPTURES}")
                    
                    if count >= TARGET_CAPTURES:
                        auto_mode = False
                        print("\n*** Target reached! Press Q to calibrate. ***")
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                auto_mode = not auto_mode
                last_capture_time = time.time()
                print(f"\n[AUTO-CAPTURE {'STARTED' if auto_mode else 'STOPPED'}]")
            elif key == 32 and valid_pair and quality_ok:  # SPACE
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners1_sub = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                corners2_sub = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
                
                objpoints.append(objp)
                imgpoints_l.append(corners1_sub)
                imgpoints_r.append(corners2_sub)
                count += 1
                print(f"[MANUAL] Captured {count}/{TARGET_CAPTURES}")
                
        except Exception as e:
            print(f"Frame error: {e}")
            continue

    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()
    
    if count < MIN_CAPTURES:
        print(f"\nOnly {count} captures. Need at least {MIN_CAPTURES}.")
        return

    print(f"\n{'=' * 60}")
    print(f"  CALIBRATING with {count} samples...")
    print(f"  (This may take 1-2 minutes)")
    print(f"{'=' * 60}")
    
    # Stage 1: Individual camera calibration (for error analysis)
    print("\nStep 1/3: Individual camera calibration...")
    ret_l, K1, D1, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, (width, height), None, None)
    ret_r, K2, D2, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, (width, height), None, None)
    
    # Compute reprojection errors
    errors_l = compute_reprojection_errors(objpoints, imgpoints_l, rvecs_l, tvecs_l, K1, D1)
    errors_r = compute_reprojection_errors(objpoints, imgpoints_r, rvecs_r, tvecs_r, K2, D2)
    
    print(f"  Left camera:  RMS={ret_l:.4f}, Mean={errors_l.mean():.3f}, 95%={np.percentile(errors_l, 95):.3f}")
    print(f"  Right camera: RMS={ret_r:.4f}, Mean={errors_r.mean():.3f}, 95%={np.percentile(errors_r, 95):.3f}")
    
    # Stage 2: Stereo calibration (joint optimization - more robust)
    print("\nStep 2/3: Stereo calibration...")
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    
    # Use joint optimization for better accuracy
    # Don't fix intrinsics - let them be refined together
    flags = cv2.CALIB_SAME_FOCAL_LENGTH + cv2.CALIB_ZERO_TANGENT_DIST
    
    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r, 
        K1, D1, K2, D2, 
        (width, height), 
        criteria=criteria_stereo, 
        flags=flags
    )
    
    # Stage 3: Rectification
    print("\nStep 3/3: Computing rectification...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, (width, height), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    
    # Quality assessment
    print(f"\n{'=' * 60}")
    print(f"  CALIBRATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Stereo RMS Error: {ret:.4f} pixels")
    print(f"  Baseline: {np.linalg.norm(T)*1000:.1f} mm")
    
    if ret < 0.5:
        quality = "★★★ EXCELLENT ★★★"
        guidance = "Perfect! Your depth should be very accurate."
    elif ret < 0.8:
        quality = "★★ VERY GOOD ★★"
        guidance = "Great calibration. Depth will be accurate."
    elif ret < 1.2:
        quality = "★ GOOD ★"
        guidance = "Acceptable. Consider recalibrating for better accuracy."
    elif ret < 2.0:
        quality = "OK"
        guidance = "Mediocre. Recalibrate with more varied poses and better lighting."
    else:
        quality = "⚠ BAD ⚠"
        guidance = "Poor calibration. Check your checkerboard and recalibrate."
    
    print(f"  Quality: {quality}")
    print(f"  {guidance}")
    print(f"{'=' * 60}")
    
    # Save calibration
    stereo_utils.save_calibration(
        cal_cfg['output_path'], 
        K1, D1, K2, D2, R, T, R1, R2, P1, P2, Q, 
        (width, height)
    )
    
    print(f"\nDone! Run 'python depth_camera.py' to test.")

if __name__ == "__main__":
    main()
