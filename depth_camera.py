import cv2
import numpy as np
import time
import config
import pointcloud
import rendering
import stereo
import display
import mouse
from camera import ThreadedCamera
from collections import deque

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False


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
    
    map1l, map2l, map1r, map2r = stereo.build_rectify_maps(K1, D1, K2, D2, R1, R2, P1, P2, (w, h))
    
    focal = K1[0, 0]
    baseline = s_cfg.get('baseline_meters', 0.08)
    auto_p = pointcloud.compute_sgbm_params(baseline, focal, d_cfg['z_min'], d_cfg['z_max'])
    
    min_d = s_cfg.get('min_disparity', auto_p['min_disparity'])
    num_d = s_cfg.get('num_disparities', auto_p['num_disparities'])
    blk = s_cfg.get('block_size', 5)
    
    stereo_l = stereo.create_sgbm(min_d, num_d, blk, s_cfg)
    stereo_r, wls = stereo.create_wls(stereo_l, s_cfg) if s_cfg.get('use_wls_filter', True) else (None, None)
    
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
        cv2.setMouseCallback("Ghost View", mouse.callback)
    
    use_tex = False
    prev_t = time.time()
    disp_buf = deque(maxlen=5)
    use_temp = True
    use_conf = True
    
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
            low_tex = display.low_texture_mask(gray_l)
        
        disp = stereo.compute_disparity(stereo_l, stereo_r, wls, gray_l, gray_r)
        
        if use_temp:
            disp_buf.append(disp)
            if len(disp_buf) > 1:
                disp = np.mean(disp_buf, axis=0)
        
        if s_cfg.get('use_median_filter', True):
            disp = pointcloud.apply_filters(disp, s_cfg.get('median_ksize', 5))
        
        if use_conf:
            disp[low_tex] = 0
        
        dc = display.colorize_depth(disp, min_d, num_d)
        
        now = time.time()
        fps = 1.0 / (now - prev_t + 1e-5)
        prev_t = now
        
        display.overlay_fps(dc, fps)
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
                    rx, ry = mouse.get_rotation()
                    ghost_img = rendering.render_cloud(pts, vc, rx=rx, ry=ry)
                    display.overlay_points(ghost_img, len(pts))
                
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
