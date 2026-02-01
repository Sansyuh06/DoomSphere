import cv2
import numpy as np
import time
import config
import pointcloud
import rendering
from camera import ThreadedCamera
from collections import deque

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

mouse = {'drag': False, 'lx': 0, 'ly': 0, 'rx': -20, 'ry': 0}

def on_mouse(event, x, y, flags, param):
    global mouse
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse['drag'] = True
        mouse['lx'], mouse['ly'] = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        mouse['drag'] = False
    elif event == cv2.EVENT_MOUSEMOVE and mouse['drag']:
        mouse['ry'] += (x - mouse['lx']) * 0.5
        mouse['rx'] += (y - mouse['ly']) * 0.5
        mouse['lx'], mouse['ly'] = x, y


def main():
    print("=" * 50)
    print("  STEREO DEPTH CAMERA")
    print("=" * 50)
    
    cfg = config.load_config()
    calib = config.load_calibration(cfg['calibration']['output_path'])
    
    if calib is None:
        print("Run calibrate_stereo.py first.")
        return
    
    s_cfg = cfg['stereo']
    d_cfg = cfg['depth']
    c_cfg = cfg['cameras']
    
    K1, D1, K2, D2 = calib['K1'], calib['D1'], calib['K2'], calib['D2']
    R, T = calib['R'], calib['T']
    w, h = calib['image_size']
    
    if 'R1' in calib:
        R1, R2, P1, P2, Q = calib['R1'], calib['R2'], calib['P1'], calib['P2'], calib['Q']
    else:
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T)
    
    map1l, map2l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_16SC2)
    map1r, map2r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_16SC2)
    
    focal = K1[0, 0]
    baseline = s_cfg.get('baseline_meters', 0.08)
    auto_p = pointcloud.compute_sgbm_params(baseline, focal, d_cfg['z_min'], d_cfg['z_max'])
    
    min_d = s_cfg.get('min_disparity', auto_p['min_disparity'])
    num_d = s_cfg.get('num_disparities', auto_p['num_disparities'])
    blk = s_cfg.get('block_size', 5)
    
    stereo_l = cv2.StereoSGBM_create(
        minDisparity=min_d, numDisparities=num_d, blockSize=blk,
        P1=8 * 3 * blk**2, P2=32 * 3 * blk**2,
        disp12MaxDiff=s_cfg.get('disp12_max_diff', 5),
        uniquenessRatio=s_cfg.get('uniqueness_ratio', 5),
        speckleWindowSize=s_cfg.get('speckle_window_size', 200),
        speckleRange=s_cfg.get('speckle_range', 2),
        preFilterCap=s_cfg.get('pre_filter_cap', 31),
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    use_wls = s_cfg.get('use_wls_filter', True)
    wls, stereo_r = None, None
    if use_wls:
        try:
            stereo_r = cv2.ximgproc.createRightMatcher(stereo_l)
            wls = cv2.ximgproc.createDisparityWLSFilter(stereo_l)
            wls.setLambda(s_cfg.get('wls_lambda', 8000))
            wls.setSigmaColor(s_cfg.get('wls_sigma', 1.2))
        except:
            use_wls = False
    
    cam_l = ThreadedCamera(c_cfg['left_id'], c_cfg['width'], c_cfg['height']).start()
    cam_r = ThreadedCamera(c_cfg['right_id'], c_cfg['width'], c_cfg['height']).start()
    time.sleep(1.0)
    
    pcd, vis = None, None
    if HAS_O3D:
        vis = o3d.visualization.Visualizer()
        vis.create_window("Ghost View", width=800, height=600)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.1])
        opt.point_size = 2.5
    else:
        cv2.namedWindow("Ghost View")
        cv2.setMouseCallback("Ghost View", on_mouse)
    
    use_tex = False
    prev_t = time.time()
    disp_buf = deque(maxlen=5)
    use_temp = True
    use_conf = True
    conf_thresh = 15.0
    
    ghost_img = np.zeros((600, 600, 3), dtype=np.uint8)
    
    while True:
        _, frame_l = cam_l.read()
        _, frame_r = cam_r.read()
        if frame_l is None or frame_r is None:
            continue
        
        rect_l = cv2.remap(frame_l, map1l, map2l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(frame_r, map1r, map2r, cv2.INTER_LINEAR)
        
        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
        
        if use_conf:
            sx = cv2.Sobel(gray_l, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray_l, cv2.CV_64F, 0, 1, ksize=3)
            low_tex = np.sqrt(sx**2 + sy**2) < conf_thresh
        
        disp_l = stereo_l.compute(gray_l, gray_r)
        
        if use_wls and wls:
            disp_r = stereo_r.compute(gray_r, gray_l)
            disp = wls.filter(disp_l, gray_l, disparity_map_right=disp_r).astype(np.float32) / 16.0
        else:
            disp = disp_l.astype(np.float32) / 16.0
        
        if use_temp:
            disp_buf.append(disp)
            if len(disp_buf) > 1:
                disp = np.mean(disp_buf, axis=0)
        
        if s_cfg.get('use_median_filter', True):
            disp = pointcloud.apply_filters(disp, s_cfg.get('median_ksize', 5))
        
        if use_conf:
            disp[low_tex] = 0
        
        dv = np.clip((disp - min_d) / num_d, 0, 1)
        dimg = (dv * 255).astype(np.uint8)
        dc = cv2.applyColorMap(dimg, cv2.COLORMAP_JET)
        
        now = time.time()
        fps = 1.0 / (now - prev_t + 1e-5)
        prev_t = now
        
        cv2.putText(dc, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Depth View", dc)
        
        colors_src = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB) if use_tex else None
        xyz = cv2.reprojectImageTo3D(disp, Q)
        z = xyz[:, :, 2]
        valid = (disp > 0) & np.isfinite(z)
        
        if np.any(valid):
            zv = z[valid]
            if np.median(zv) < 0:
                z = -z
                xyz[:, :, 2] = z
            
            mask = valid & (z > d_cfg['z_min']) & (z < d_cfg['z_max'])
            pts = xyz[mask]
            pt_colors = colors_src[mask] if colors_src is not None else None
            
            if len(pts) > 0:
                if use_tex and pt_colors is not None:
                    vc = pt_colors / 255.0
                else:
                    vc = rendering.depth_colors(pts[:, 2], d_cfg['z_min'], d_cfg['z_max'])
                
                max_pts = d_cfg.get('max_points', 50000)
                if len(pts) > max_pts:
                    idx = np.linspace(0, len(pts) - 1, max_pts).astype(int)
                    pts, vc = pts[idx], vc[idx]
                
                if HAS_O3D:
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    pcd.colors = o3d.utility.Vector3dVector(vc)
                    vis.update_geometry(pcd)
                    vis.poll_events()
                    vis.update_renderer()
                else:
                    ghost_img = rendering.render_cloud(pts, vc, rx=mouse['rx'], ry=mouse['ry'])
                    cv2.putText(ghost_img, f"Pts: {len(pts)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                map_img = rendering.render_topdown(pts)
                cv2.imshow("Map", map_img)
        
        cv2.imshow("Ghost View", ghost_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            use_tex = not use_tex
        elif key == ord('a'):
            use_temp = not use_temp
            disp_buf.clear()
        elif key == ord('c'):
            use_conf = not use_conf
        elif key == ord('s'):
            cv2.imwrite("depth.png", dc)
    
    cam_l.stop()
    cam_r.stop()
    cv2.destroyAllWindows()
    if HAS_O3D:
        vis.destroy_window()


if __name__ == "__main__":
    main()
