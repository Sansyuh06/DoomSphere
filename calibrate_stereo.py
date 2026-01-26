import cv2
import numpy as np
import time
import stereo_utils

TARGET_CAPTURES = 55  # Number of calibration samples to collect

def main():
    print("=" * 50)
    print("  STEREO CALIBRATION TOOL (Auto-Capture)")
    print("=" * 50)
    
    config = stereo_utils.load_config()
    
    # Setup
    c_cfg = config['cameras']
    cal_cfg = config['calibration']
    
    BOARD_SIZE = tuple(cal_cfg['chessboard_size'])
    SQUARE_SIZE = cal_cfg['square_size_meters']
    
    # Prepare object points
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints_l = []
    imgpoints_r = []

    # Open Cameras
    print(f"Opening cameras {c_cfg['left_id']} and {c_cfg['right_id']}...")
    cam1 = cv2.VideoCapture(c_cfg['left_id'], cv2.CAP_DSHOW)
    cam2 = cv2.VideoCapture(c_cfg['right_id'], cv2.CAP_DSHOW)
    
    width, height = c_cfg['width'], c_cfg['height']
    cam1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cam1.isOpened() or not cam2.isOpened():
        print("Error: Could not open cameras.")
        return

    print("\nControls:")
    print("  'C' - Start/Stop AUTO-CAPTURE")
    print("  'SPACE' - Manual capture (single frame)")
    print("  'Q' - Finish and Calibrate")
    print(f"\nTarget: {TARGET_CAPTURES} captures")
    print("-" * 50)
    
    count = 0
    auto_mode = False
    last_capture_time = 0
    CAPTURE_INTERVAL = 1.5  # Seconds between auto-captures
    
    while True:
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        
        if not ret1 or not ret2:
            continue

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Find corners (with fast check for performance)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret_c1, corners1 = cv2.findChessboardCorners(gray1, BOARD_SIZE, flags)
        ret_c2, corners2 = cv2.findChessboardCorners(gray2, BOARD_SIZE, flags)
        
        vis1 = frame1.copy()
        vis2 = frame2.copy()
        
        valid_pair = ret_c1 and ret_c2
        
        # Draw detection feedback
        if valid_pair:
            cv2.drawChessboardCorners(vis1, BOARD_SIZE, corners1, ret_c1)
            cv2.drawChessboardCorners(vis2, BOARD_SIZE, corners2, ret_c2)
            cv2.putText(vis1, "DETECTED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(vis1, "Looking for board...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Status display
        cv2.putText(vis1, f"Captures: {count}/{TARGET_CAPTURES}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if auto_mode:
            cv2.putText(vis1, "AUTO MODE ON", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Progress bar
        progress = int((count / TARGET_CAPTURES) * (width - 20))
        cv2.rectangle(vis1, (10, height - 30), (10 + progress, height - 10), (0, 255, 0), -1)
        cv2.rectangle(vis1, (10, height - 30), (width - 10, height - 10), (255, 255, 255), 2)

        # Combined view
        combined = np.hstack((vis1, vis2))
        cv2.imshow('Stereo Calibration (Press C for Auto)', combined)
        
        # Auto-capture logic
        if auto_mode and valid_pair and count < TARGET_CAPTURES:
            current_time = time.time()
            if current_time - last_capture_time >= CAPTURE_INTERVAL:
                # Refine corners
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
                    print("\n*** Target reached! Press 'Q' to calibrate. ***")
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            auto_mode = not auto_mode
            last_capture_time = time.time()
            if auto_mode:
                print("\n[AUTO-CAPTURE STARTED] Move the board around slowly...")
            else:
                print("\n[AUTO-CAPTURE STOPPED]")
        elif key == 32:  # SPACE for manual capture
            if valid_pair:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners1_sub = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                corners2_sub = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
                
                objpoints.append(objp)
                imgpoints_l.append(corners1_sub)
                imgpoints_r.append(corners2_sub)
                count += 1
                print(f"[MANUAL] Captured {count}/{TARGET_CAPTURES}")

    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()
    
    if count < 15:
        print(f"\nOnly {count} captures. Need at least 15 for good calibration. Exiting.")
        return

    print(f"\n{'=' * 50}")
    print(f"  CALIBRATING with {count} samples...")
    print(f"  (This may take 1-2 minutes)")
    print(f"{'=' * 50}")
    
    # Stereo Calibration
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    flags = cv2.CALIB_SAME_FOCAL_LENGTH + cv2.CALIB_ZERO_TANGENT_DIST
    
    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r, 
        None, None, None, None, 
        (width, height), 
        criteria=criteria_stereo, 
        flags=flags
    )
    
    print(f"\n{'=' * 50}")
    print(f"  RMS Error: {ret:.4f}")
    if ret < 0.5:
        print("  Status: ★★★ EXCELLENT ★★★")
    elif ret < 1.0:
        print("  Status: ★★ VERY GOOD ★★")
    elif ret < 1.5:
        print("  Status: ★ GOOD ★")
    elif ret < 3.0:
        print("  Status: OK (Consider recalibrating)")
    else:
        print("  Status: BAD (Please recalibrate)")
    print(f"{'=' * 50}")

    # Rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, (width, height), R, T
    )
    
    # --- VISUAL VERIFICATION ---
    print("\n[VERIFY] Checking rectification...")
    # Grab a fresh pair
    while True:
        if cam1.isOpened(): ret1, f1 = cam1.read()
        else: f1 = None
        if cam2.isOpened(): ret2, f2 = cam2.read()
        else: f2 = None
        if ret1 and ret2: break
        time.sleep(0.1)

    # Undistort and Rectify
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)
    
    rect1 = cv2.remap(f1, map1x, map1y, cv2.INTER_LINEAR)
    rect2 = cv2.remap(f2, map2x, map2y, cv2.INTER_LINEAR)
    
    # Draw horizontal green lines every 30 pixels
    vis_rect = np.hstack((rect1, rect2))
    for y in range(0, height, 30):
        cv2.line(vis_rect, (0, y), (width*2, y), (0, 255, 0), 1)
        
    cv2.imshow("Rectification Check (Press Space to Save, Esc to Discard)", vis_rect)
    key = cv2.waitKey(0)
    
    if key == 27: # ESC
        print("Calibration discarded.")
    else:
        # Save full data
        stereo_utils.save_calibration(cal_cfg['output_path'], K1, D1, K2, D2, R, T, R1, R2, P1, P2, Q, (width, height))
        print(f"\nDone! Run 'python depth_camera.py' to see the results.")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
