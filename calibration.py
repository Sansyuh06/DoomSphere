import cv2
import numpy as np


def find_corners(gray, board_size):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    return cv2.findChessboardCorners(gray, board_size, flags)


def refine_corners(gray, corners):
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)


def check_quality(corners, w, h, min_area_ratio=0.05, edge_margin=20):
    hull = cv2.convexHull(corners)
    area = cv2.contourArea(hull)
    
    if area < w * h * min_area_ratio:
        return False
    
    pts = corners.reshape(-1, 2)
    if (np.any(pts[:, 0] < edge_margin) or np.any(pts[:, 0] > w - edge_margin) or
        np.any(pts[:, 1] < edge_margin) or np.any(pts[:, 1] > h - edge_margin)):
        return False
    
    return True


def calc_reproj_errors(objpts, imgpts, rvecs, tvecs, K, D):
    all_errs = []
    for i in range(len(objpts)):
        proj, _ = cv2.projectPoints(objpts[i], rvecs[i], tvecs[i], K, D)
        err = np.linalg.norm(imgpts[i].reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
        all_errs.extend(err)
    return np.array(all_errs)


def build_object_points(board_size, square_size):
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp


def calibrate_stereo(objpoints, imgpts_l, imgpts_r, size):
    w, h = size
    
    ret_l, K1, D1, rv_l, tv_l = cv2.calibrateCamera(objpoints, imgpts_l, (w, h), None, None)
    ret_r, K2, D2, rv_r, tv_r = cv2.calibrateCamera(objpoints, imgpts_r, (w, h), None, None)
    
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    flags = cv2.CALIB_SAME_FOCAL_LENGTH + cv2.CALIB_ZERO_TANGENT_DIST
    
    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpts_l, imgpts_r, K1, D1, K2, D2, (w, h), 
        criteria=crit, flags=flags
    )
    
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, (w, h), R, T, 
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    
    return ret, K1, D1, K2, D2, R, T, R1, R2, P1, P2, Q


def quality_rating(rms):
    if rms < 0.5:
        return "EXCELLENT"
    elif rms < 1.0:
        return "GOOD"
    elif rms < 2.0:
        return "OK"
    return "BAD"
