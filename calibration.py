import cv2
import numpy as np


def find_corners(gray, board_size):
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + 
             cv2.CALIB_CB_NORMALIZE_IMAGE + 
             cv2.CALIB_CB_FAST_CHECK)
    
    found, corners = cv2.findChessboardCorners(gray, board_size, flags)
    
    if not found:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        found, corners = cv2.findChessboardCorners(enhanced, board_size, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    return found, corners


def refine_corners(gray, corners):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def check_quality(corners, w, h, min_area_ratio=0.02, edge_margin=10):
    hull = cv2.convexHull(corners)
    area = cv2.contourArea(hull)
    
    if area < w * h * min_area_ratio:
        return False
    
    pts = corners.reshape(-1, 2)
    if (np.any(pts[:, 0] < edge_margin) or np.any(pts[:, 0] > w - edge_margin) or
        np.any(pts[:, 1] < edge_margin) or np.any(pts[:, 1] > h - edge_margin)):
        return False
    
    return True


def build_object_points(board_size, square_size):
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp


def calibrate_stereo(objpoints, imgpts_l, imgpts_r, img_size):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    
    ret1, K1, D1, _, _ = cv2.calibrateCamera(objpoints, imgpts_l, img_size, None, None)
    ret2, K2, D2, _, _ = cv2.calibrateCamera(objpoints, imgpts_r, img_size, None, None)
    
    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpts_l, imgpts_r,
        K1, D1, K2, D2, img_size,
        criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC
    )
    
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    
    return ret, K1, D1, K2, D2, R, T, R1, R2, P1, P2, Q


def quality_rating(rms):
    if rms < 0.3:
        return "EXCELLENT"
    elif rms < 0.5:
        return "GOOD"
    elif rms < 1.0:
        return "OK"
    else:
        return "POOR"
