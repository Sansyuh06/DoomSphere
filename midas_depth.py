import cv2
import numpy as np
import time
import sys
import os
from rendering import render_cloud

print("Initializing MiDaS...", flush=True)

try:
    import torch
    import torch.hub
    HAS_TORCH = True
    print("PyTorch loaded.", flush=True)
except ImportError as e:
    HAS_TORCH = False
    print(f"PyTorch not found: {e}", flush=True)
except Exception as e:
    HAS_TORCH = False
    print(f"Error: {e}", flush=True)

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
    print("  MiDaS AI DEPTH")
    print("=" * 50)
    
    if not HAS_TORCH:
        print("PyTorch required: pip install torch torchvision timm")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("Loading model...", flush=True)
    try:
        model_type = "MiDaS_small"
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.to(device)
        midas.eval()
        
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = transforms.small_transform if model_type == "MiDaS_small" else transforms.dpt_transform
        print(f"Loaded: {model_type}")
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
    cv2.setMouseCallback("Ghost", on_mouse)
    
    print("Q=Quit, S=Save")
    
    last_t = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_t = transform(rgb).to(device)
        
        with torch.no_grad():
            pred = midas(input_t)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=frame.shape[:2], mode="bicubic", align_corners=False
            ).squeeze()
        
        depth = pred.cpu().numpy()
        
        d_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-5)
        d_vis = (d_norm * 255).astype(np.uint8)
        dc = cv2.applyColorMap(d_vis, cv2.COLORMAP_JET)
        
        now = time.time()
        fps = 1.0 / (now - last_t + 1e-5)
        last_t = now
        
        cv2.putText(dc, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
            ghost = render_cloud(pts, cloud_colors, rx=mouse['rx'], ry=mouse['ry'])
            cv2.putText(ghost, f"Pts: {len(pts)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
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
