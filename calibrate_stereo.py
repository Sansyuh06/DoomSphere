import cv2
import numpy as np
import time
import config
import calibration as calib
from camera import open_camera


def main():
    print("=" * 50)
    print("  STEREO CALIBRATION")
    print("=" * 50)
    
    cfg = config.load_config()
    cam_cfg = cfg['cameras']
    cal_cfg = cfg['calibration']
    
    BOARD = tuple(cal_cfg['chessboard_size'])
    SQUARE = cal_cfg['square_size_meters']
    TARGET = cal_cfg.get('target_captures', 55)
    MIN_CAPS = cal_cfg.get('min_captures', 30)
    
    objp = calib.build_object_points(BOARD, SQUARE)
    objpoints, imgpts_left, imgpts_right = [], [], []

    w, h = cam_cfg['width'], cam_cfg['height']
    cam1 = open_camera(cam_cfg['left_id'], w, h)
    cam2 = open_camera(cam_cfg['right_id'], w, h)

    if cam1 is None or cam2 is None:
        print("ERROR: Could not open cameras!")
        return

    print(f"Board: {BOARD[0]}x{BOARD[1]}, {SQUARE*1000:.1f}mm")
    print("C=Auto, SPACE=Manual, Q=Done")
    
    count = 0
    auto_mode = False
    last_cap = 0
    
    while True:
        try:
            ret1, frame1 = cam1.read()
            ret2, frame2 = cam2.read()
            if not ret1 or not ret2:
                continue
            
            if frame1.shape[:2] != (h, w):
                frame1 = cv2.resize(frame1, (w, h))
            if frame2.shape[:2] != (h, w):
                frame2 = cv2.resize(frame2, (w, h))

            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
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
            color = (0, 255, 0) if (both and ok) else (0, 0, 255)
            cv2.putText(vis1, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(vis1, f"{count}/{TARGET}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
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

    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()
    
    if count < MIN_CAPS:
        print(f"Need at least {MIN_CAPS} captures.")
        return

    print(f"\nCalibrating with {count} samples...")
    
    ret, K1, D1, K2, D2, R, T, R1, R2, P1, P2, Q = calib.calibrate_stereo(
        objpoints, imgpts_left, imgpts_right, (w, h)
    )
    
    print(f"\nRMS: {ret:.4f} px, Baseline: {np.linalg.norm(T)*1000:.1f} mm")
    print(f"Quality: {calib.quality_rating(ret)}")
    
    config.save_calibration(cal_cfg['output_path'], K1, D1, K2, D2, R, T, R1, R2, P1, P2, Q, (w, h))
    print("Done!")


if __name__ == "__main__":
    main()
