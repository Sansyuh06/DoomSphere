import cv2
import numpy as np


def low_texture_mask(gray, thresh=15.0):
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(sx**2 + sy**2) < thresh


def colorize_depth(disp, min_d, num_d):
    dv = np.clip((disp - min_d) / num_d, 0, 1)
    dimg = (dv * 255).astype(np.uint8)
    return cv2.applyColorMap(dimg, cv2.COLORMAP_JET)


def overlay_fps(img, fps):
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def overlay_points(img, count):
    cv2.putText(img, f"Pts: {count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
