import cv2
import numpy as np
import stereo_utils
import sys

logger = stereo_utils.setup_logging("CalibrateStereo")

# =======================================================
# =======================================================
# CONFIGURATION LOADED FROM config.json
# =======================================================

def main():
    try:
        config = stereo_utils.load_config()
    except Exception:
        return

    # Extract config
    CHESSBOARD_SIZE = tuple(config["calibration"]["chessboard_size"])
    SQUARE_SIZE = config["calibration"]["square_size_meters"]
    CAM_ID_LEFT = config["camera"]["left_id"]
    CAM_ID_RIGHT = config["camera"]["right_id"]
    FRAME_WIDTH = config["camera"]["width"]
    FRAME_HEIGHT = config["camera"]["height"]
    OUTPUT_FILE = config["calibration"]["output_file"]
    MIN_FRAMES = config["calibration"]["min_frames"]

    logger.info("==============================================")
    logger.info(" STEREO CAMERA CALIBRATION")
    logger.info("==============================================")
    logger.info(f"Loaded Config: Left={CAM_ID_LEFT}, Right={CAM_ID_RIGHT}, Res={FRAME_WIDTH}x{FRAME_HEIGHT}")
    logger.info(f"1. Ensure both cameras are connected ({CAM_ID_LEFT}, {CAM_ID_RIGHT}).")
    logger.info(f"2. Use a chessboard with {CHESSBOARD_SIZE} inner corners.")
    logger.info("3. Keys:")
    logger.info("   'c' - Capture frame (hold board still)")
    logger.info("   'q' - Finish capturing and run calibration")
    logger.info("==============================================")

    # Open cameras
    try:
        cap_left = cv2.VideoCapture(CAM_ID_LEFT)
        cap_right = cv2.VideoCapture(CAM_ID_RIGHT)
        if not cap_left.isOpened() or not cap_right.isOpened():
             raise IOError("Cannot open one of the cameras.")
    except Exception as e:
        logger.error(f"Failed to initialize cameras: {e}")
        return

    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # Arrays to store object points and image points from all the images.
    objpoints = []   # 3d point in real world space
    imgpoints_l = [] # 2d points in image plane.
    imgpoints_r = [] # 2d points in image plane.

    count = 0

    while True:
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()

        if not ret_l or not ret_r:
            logger.warning("Failed to grab frames")
            break

        # Force resize in case hardware ignores set()
        frame_l = cv2.resize(frame_l, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_r = cv2.resize(frame_r, (FRAME_WIDTH, FRAME_HEIGHT))

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        # Visualization
        display_l = frame_l.copy()
        display_r = frame_r.copy()

        # Find corners for visualization (optional, makes it slow if done every frame)
        # We only strictly need it when 'c' is pressed, but showing it helps aiming.
        
        cv2.imshow("Left (Press 'c' to capture, 'q' to calibrate)", display_l)
        cv2.imshow("Right", display_r)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            logger.info("Searching for chessboard...")
            ret_cb_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD_SIZE, None)
            ret_cb_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD_SIZE, None)

            if ret_cb_l and ret_cb_r:
                # Refine corners
                corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1),
                                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1),
                                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

                objpoints.append(objp)
                imgpoints_l.append(corners_l)
                imgpoints_r.append(corners_r)

                count += 1
                logger.info(f"Captured Set #{count}")
                
                # Draw just to show success flash
                cv2.drawChessboardCorners(display_l, CHESSBOARD_SIZE, corners_l, ret_cb_l)
                cv2.drawChessboardCorners(display_r, CHESSBOARD_SIZE, corners_r, ret_cb_r)
                cv2.imshow("Left (Press 'c' to capture, 'q' to calibrate)", display_l)
                cv2.imshow("Right", display_r)
                cv2.waitKey(500)
            else:
                logger.warning("Chessboard not found in both images!")

        elif key == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

    if count < MIN_FRAMES:
        logger.warning(f"You have {count} calibration sets. Recommended: {MIN_FRAMES}+. Results may be poor.")

    logger.info("Calibrating cameras... This might take a moment.")
    
    # 1. Individual Calibration (Optional but good for initial guess)
    ret_l, K1, D1, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, gray_l.shape[::-1], None, None)
    ret_r, K2, D2, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, gray_r.shape[::-1], None, None)

    # 2. Stereo Calibration
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC # Fix K matrix if you want, or let it refine
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    
    ret_s, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        K1, D1, K2, D2,
        gray_l.shape[::-1], criteria=criteria, flags=flags)

    logger.info(f"Stereo Calibration RMS: {ret_s}")
    
    # 3. Rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, gray_l.shape[::-1], R, T)

    stereo_utils.save_stereo_coefficients(
        OUTPUT_FILE, K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q, gray_l.shape[::-1]
    )

if __name__ == "__main__":
    main()
