import cv2
import threading
import time


class ThreadedCamera:
    def __init__(self, src, width=640, height=480):
        self.src = src
        self.cap = None
        self.ret = False
        self.frame = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.fail_count = 0
        
        # DSHOW only on windows - most reliable for USB cams
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        time.sleep(0.3)
        
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # try a few reads to let camera warm up
            for _ in range(5):
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.ret = True
                    self.frame = frame
                    print(f"Camera {src}: OK")
                    return
                time.sleep(0.1)
            self.cap.release()
        
        print(f"Camera {src}: FAILED")
        self.cap = None

    def start(self):
        if self.cap is None:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        while self.running and self.cap:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.ret = True
                        self.frame = frame
                        self.fail_count = 0
                else:
                    self.fail_count += 1
                    if self.fail_count > 50:
                        print(f"Camera {self.src}: lost connection")
                        self.running = False
                        break
                    time.sleep(0.03)
            except:
                time.sleep(0.03)
            time.sleep(0.001)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None

    def is_ok(self):
        return self.cap is not None and self.cap.isOpened()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()


def open_camera(cam_id, w, h):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    time.sleep(0.3)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Camera {cam_id}: OK")
                return cap
            time.sleep(0.1)
        cap.release()
    print(f"Camera {cam_id}: FAILED")
    return None
