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
        print("1. Start (Calibrate + Depth)")
        print("2. Recalibrate")
        print("3. Exit")
        
        c = input("\nChoice: ").strip()
        
        if c == '1':
            run("main.py")
        elif c == '2':
            if os.path.exists("stereo_params.npz"):
                os.remove("stereo_params.npz")
                print("Old calibration removed.")
            run("main.py")
        elif c == '3':
            break
        else:
            time.sleep(0.3)


if __name__ == "__main__":
    main()
