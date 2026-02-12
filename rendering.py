import cv2
import numpy as np


def render_cloud(pts, colors, w=600, h=600, rx=0, ry=0, center=None):
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if len(pts) == 0:
        return canvas, center
    
    if center is None:
        center = np.median(pts, axis=0)
    else:
        new_center = np.median(pts, axis=0)
        center = 0.1 * new_center + 0.9 * center
    
    p = pts - center
    spread = np.percentile(np.abs(p), 90) + 0.01
    scale = 2.0 / spread
    p *= scale
    
    ay = np.radians(ry)
    cy, sy = np.cos(ay), np.sin(ay)
    x = p[:, 0] * cy + p[:, 2] * sy
    z = -p[:, 0] * sy + p[:, 2] * cy
    y = p[:, 1]
    
    ax = np.radians(rx)
    cx, sx = np.cos(ax), np.sin(ax)
    y2 = y * cx - z * sx
    z2 = y * sx + z * cx
    
    f = 400
    zp = z2 + 4.0
    ok = zp > 0.2
    
    u = (x[ok] * f / zp[ok] + w // 2).astype(int)
    v = (y2[ok] * f / zp[ok] + h // 2).astype(int)
    c = colors[ok]
    
    good = (u >= 1) & (u < w - 2) & (v >= 1) & (v < h - 2)
    u = u[good]
    v = v[good]
    c = (c[good] * 255).astype(np.uint8)
    
    depth_order = np.argsort(-zp[ok][good])
    u = u[depth_order]
    v = v[depth_order]
    c = c[depth_order]
    
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            canvas[v + dy, u + dx] = c
    
    return canvas, center


def render_topdown(pts, w=400, h=400, x_range=(-2.0, 2.0), z_range=(0.0, 5.0)):
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if len(pts) == 0:
        return canvas
    
    mask = ((pts[:, 0] > x_range[0]) & (pts[:, 0] < x_range[1]) &
            (pts[:, 2] > z_range[0]) & (pts[:, 2] < z_range[1]))
    p = pts[mask]
    if len(p) == 0:
        return canvas
    
    u = ((p[:, 0] - x_range[0]) / (x_range[1] - x_range[0]) * w).astype(int)
    v = ((1.0 - (p[:, 2] - z_range[0]) / (z_range[1] - z_range[0])) * h).astype(int)
    
    u = np.clip(u, 0, w - 1)
    v = np.clip(v, 0, h - 1)
    
    ht = -p[:, 1]
    hn = np.clip((ht + 0.5) / 1.0, 0, 1)
    c = np.zeros((len(ht), 3), dtype=np.uint8)
    c[:, 0] = (255 * (1 - hn)).astype(np.uint8)
    c[:, 2] = (255 * hn).astype(np.uint8)
    
    canvas[v, u] = c
    return canvas


def depth_colors(z, z_min, z_max):
    norm = np.clip((z - z_min) / (z_max - z_min + 1e-5), 0, 1)
    c = np.zeros((len(z), 3))
    c[:, 0] = 0.2 * (1 - norm)
    c[:, 1] = 0.8 * (1 - norm)
    c[:, 2] = 1.0 - 0.3 * norm
    return c
