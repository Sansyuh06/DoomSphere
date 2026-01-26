import cv2
import numpy as np
import os
import threading
import time
import logging
import json

# Configure logging

# Configure logging
def setup_logging(name="DoomSphere"):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(name)

def load_config(path="config.json"):
    """Load configuration from a JSON file."""
    try:
        with open(path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Config file {path} not found. Using defaults could be dangerous.")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse {path}: {e}")
        raise

logger = setup_logging()

logger = setup_logging()

class ThreadedCamera:
    """
    Performance Optimization Pattern: Threaded I/O.
    Constantly reads frames in a background thread so the main thread 
    gets the most recent frame instantly without blocking.
    """
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        # Force DirectShow for Windows to prevent black screen/connection issues
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.width = width
        self.height = height

    def start(self):
        if self.started:
            logger.warning("ThreadedCamera already started.")
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True # Thread dies when main dies
        self.thread.start()
        logger.info(f"Camera {self.src} thread started.")
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                if grabbed:
                    # Resize here in thread to offload main thread
                    # self.frame = cv2.resize(frame, (self.width, self.height))
                    # Actually, better to resize only if needed, but let's keep it raw for speed
                    self.frame = frame
            time.sleep(0.005) # Avoid killing the CPU

    def read(self):
        with self.read_lock:
            if not self.grabbed:
                return False, None
            # Return a copy to avoid race conditions if processing is slow
            return True, self.frame.copy()

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()

def save_stereo_coefficients(path, K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q, image_size):
    """ Save the stereo coefficients to an NPZ file. """
    try:
        np.savez(path, K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, E=E, F=F, 
                 R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, image_size=image_size)
    except Exception as e:
        logger.error(f"Failed to save coefficients to {path}: {e}")
        raise

    logger.info(f"Calibration data saved to {path}")

def load_stereo_coefficients(path):
    """ Load the stereo coefficients from an NPZ file. """
    try:
        data = np.load(path)
        logger.info(f"Loaded stereo coefficients from {path}")
        return data
    except FileNotFoundError:
        logger.error(f"Calibration file not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load coefficients: {e}")
        raise

def get_3d_points(disparity, Q, scaling_factor=1.0):
    """
    Reproject image disparity to 3D space.
    input: disparity map, Q matrix
    output: points (N,3), colors (N,3) (if generic)
    """
    # reprojectImageTo3D returns (H, W, 3) matrix of (x, y, z)
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    return points_3d * scaling_factor

def write_ply(fn, verts, colors):
    """ 
    Write point cloud to PLY file.
    verts: (N, 3) array of floats
    colors: (N, 3) array of uint8
    """
    ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
    logger.info(f"Saved point cloud to {fn} ({len(verts)} points)")
