import cv2
import numpy as np


def create_sgbm(min_d, num_d, blk, s_cfg):
    return cv2.StereoSGBM_create(
        minDisparity=min_d, 
        numDisparities=num_d, 
        blockSize=blk,
        P1=8 * 3 * blk**2, 
        P2=32 * 3 * blk**2,
        disp12MaxDiff=s_cfg.get('disp12_max_diff', 1),
        uniquenessRatio=s_cfg.get('uniqueness_ratio', 10),
        speckleWindowSize=s_cfg.get('speckle_window_size', 200),
        speckleRange=s_cfg.get('speckle_range', 1),
        preFilterCap=s_cfg.get('pre_filter_cap', 63),
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )


def create_wls(stereo_l, s_cfg):
    try:
        stereo_r = cv2.ximgproc.createRightMatcher(stereo_l)
        wls = cv2.ximgproc.createDisparityWLSFilter(stereo_l)
        wls.setLambda(s_cfg.get('wls_lambda', 12000))
        wls.setSigmaColor(s_cfg.get('wls_sigma', 1.5))
        return stereo_r, wls
    except:
        return None, None


def preprocess(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def compute_disparity(stereo_l, stereo_r, wls, gray_l, gray_r):
    eq_l = preprocess(gray_l)
    eq_r = preprocess(gray_r)
    
    disp_l = stereo_l.compute(eq_l, eq_r)
    
    if wls and stereo_r:
        disp_r = stereo_r.compute(eq_r, eq_l)
        disp = wls.filter(disp_l, gray_l, disparity_map_right=disp_r)
        return disp.astype(np.float32) / 16.0
    
    return disp_l.astype(np.float32) / 16.0


def build_rectify_maps(K1, D1, K2, D2, R1, R2, P1, P2, size):
    map1l, map2l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_16SC2)
    map1r, map2r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_16SC2)
    return map1l, map2l, map1r, map2r
