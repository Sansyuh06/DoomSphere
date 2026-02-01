import cv2
import numpy as np
import time
import os
import mouse
import display
from rendering import render_cloud

print("Initializing MiDaS...", flush=True)

try:
    import midas as midas_model
    HAS_TORCH = True
    print("PyTorch loaded.", flush=True)
except ImportError as e:
    HAS_TORCH = False
    print(f"PyTorch not found: {e}", flush=True)


def main():
    print("=" * 50)
    print("  MiDaS AI DEPTH")
    print("=" * 50)
    
    if not HAS_TORCH:
        print("PyTorch required: pip install torch torchvision timm")
        return
    
    print("Loading model...", flush=True)
    try:
        model, transform, device = midas_model.load_model()
        print(f"Device: {device}")
    except Exception as e:
        print(f"Error: {e}", flush=True)
        return
    
    print("Finding camera...")
    cap = None
    backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
    
    for cam_id in [0, 1, 2, 3]:
        temp = cv2.VideoCapture(cam_id, backend)
        if temp.isOpened() and temp.read()[0]:
            cap = temp
            break
        temp.release()
    
    if cap is None:
        for cam_id in [0, 1, 2, 3]:
            temp = cv2.VideoCapture(cam_id)
            if temp.isOpened() and temp.read()[0]:
                cap = temp
                break
            temp.release()
    
    if cap is None:
        print("No camera found!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    cv2.namedWindow("Depth")
    cv2.namedWindow("Ghost")
    cv2.setMouseCallback("Ghost", mouse.callback)
    
    print("Q=Quit, S=Save")
    
    last_t = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        depth = midas_model.infer(model, transform, frame, device)
        
        d_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-5)
        d_vis = (d_norm * 255).astype(np.uint8)
        dc = cv2.applyColorMap(d_vis, cv2.COLORMAP_JET)
        
        now = time.time()
        fps = 1.0 / (now - last_t + 1e-5)
        last_t = now
        
        display.overlay_fps(dc, fps)
        cv2.imshow("Depth", dc)
        
        h, w = depth.shape
        fx = fy = w
        cx, cy = w / 2, h / 2
        
        uu, vv = np.meshgrid(np.arange(w), np.arange(h))
        z = 1.0 / (depth + 0.01)
        z = (z - z.min()) / (z.max() - z.min() + 1e-5)
        
        xx = (uu - cx) * z / fx
        yy = (vv - cy) * z / fy
        
        step = 4
        pts = np.stack([
            xx[::step, ::step].flatten(),
            yy[::step, ::step].flatten(),
            z[::step, ::step].flatten()
        ], axis=1)
        
        img_colors = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[::step, ::step].reshape(-1, 3) / 255.0
        
        valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0.05) & (pts[:, 2] < 0.95)
        pts = pts[valid]
        cloud_colors = img_colors[valid]
        
        if len(pts) > 0:
            rx, ry = mouse.get_rotation()
            ghost = render_cloud(pts, cloud_colors, rx=rx, ry=ry)
            display.overlay_points(ghost, len(pts))
            cv2.imshow("Ghost", ghost)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("midas_depth.png", dc)
            cv2.imwrite("midas_color.png", frame)
            print("Saved!")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
