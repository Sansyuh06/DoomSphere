import cv2
import time
import calibration as calib
from camera import open_camera

print("Chessboard Detection Test (Enhanced)")
print("Hold your chessboard in front of BOTH cameras")
print("Press Q to quit\n")

cam1 = open_camera(1, 640, 480)
time.sleep(1.5)
cam2 = open_camera(2, 640, 480)

if cam1 is None or cam2 is None:
    print("Camera failed!")
    exit()

sizes = [(7,7), (6,6), (8,8), (9,6), (6,9), (7,6), (6,7), (5,5), (8,5), (5,8)]

while True:
    r1, f1 = cam1.read()
    r2, f2 = cam2.read()
    if not r1 or not r2:
        continue
    
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    
    vis1, vis2 = f1.copy(), f2.copy()
    detected = None
    
    for sz in sizes:
        ok1, c1 = calib.find_corners(g1, sz)
        ok2, c2 = calib.find_corners(g2, sz)
        
        if ok1:
            cv2.drawChessboardCorners(vis1, sz, c1, ok1)
        if ok2:
            cv2.drawChessboardCorners(vis2, sz, c2, ok2)
        
        if ok1 and ok2:
            detected = sz
            break
        elif ok1 or ok2:
            detected = sz
            break
    
    if detected:
        txt = f"FOUND: {detected[0]}x{detected[1]}"
        cv2.putText(vis1, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(vis1, "No board found - try tilting", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    combined = cv2.hconcat([vis1, vis2])
    cv2.imshow("Chess Test", combined)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam1.release()
cam2.release()
cv2.destroyAllWindows()

if detected:
    print(f"\nBoard: {detected[0]}x{detected[1]} inner corners")
    print(f"Update config.json chessboard_size to [{detected[0]}, {detected[1]}]")
else:
    print("\nBoard not detected. The plastic glare may be too strong.")
    print("Try removing the plastic cover or reducing light reflections.")
