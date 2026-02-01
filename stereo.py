import cv2


def create_sgbm(min_d, num_d, blk, s_cfg):
    return cv2.StereoSGBM_create(
        minDisparity=min_d, 
        numDisparities=num_d, 
        blockSize=blk,
        P1=8 * 3 * blk**2, 
        P2=32 * 3 * blk**2,
        disp12MaxDiff=s_cfg.get('disp12_max_diff', 5),
        uniquenessRatio=s_cfg.get('uniqueness_ratio', 5),
        speckleWindowSize=s_cfg.get('speckle_window_size', 200),
        speckleRange=s_cfg.get('speckle_range', 2),
        preFilterCap=s_cfg.get('pre_filter_cap', 31),
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )


def create_wls(stereo_l, s_cfg):
    try:
        stereo_r = cv2.ximgproc.createRightMatcher(stereo_l)
        wls = cv2.ximgproc.createDisparityWLSFilter(stereo_l)
        wls.setLambda(s_cfg.get('wls_lambda', 8000))
        wls.setSigmaColor(s_cfg.get('wls_sigma', 1.2))
        return stereo_r, wls
    except:
        return None, None


def compute_disparity(stereo_l, stereo_r, wls, gray_l, gray_r):
    disp_l = stereo_l.compute(gray_l, gray_r)
    
    if wls and stereo_r:
        disp_r = stereo_r.compute(gray_r, gray_l)
        disp = wls.filter(disp_l, gray_l, disparity_map_right=disp_r)
        return disp.astype(float) / 16.0
    
    return disp_l.astype(float) / 16.0


def build_rectify_maps(K1, D1, K2, D2, R1, R2, P1, P2, size):
    map1l, map2l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_16SC2)
    map1r, map2r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_16SC2)
    return map1l, map2l, map1r, map2r
