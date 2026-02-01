import os
import sys
import subprocess
import time


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def run(script):
    if not os.path.exists(script):
        print(f"{script} not found!")
        input("Enter...")
        return
    try:
        subprocess.run([sys.executable, script])
    except KeyboardInterrupt:
        pass
    input("Enter to return...")


def main():
    while True:
        clear()
        print("=" * 40)
        print("  DOOMSPHERE")
        print("=" * 40)
        print("1. Stereo Depth")
        print("2. MiDaS AI Depth")
        print("3. Calibration")
        print("4. Exit")
        
        c = input("\nChoice: ").strip()
        
        if c == '1':
            run("depth_camera.py")
        elif c == '2':
            run("midas_depth.py")
        elif c == '3':
            run("calibrate_stereo.py")
        elif c == '4':
            break
        else:
            time.sleep(0.3)


if __name__ == "__main__":
    main()
