"""
DoomSphere Unified Launcher
===========================
Main entry point for the DoomSphere Depth System.
Allows easy switching between Stereo, AI Depth, and Calibration modes.
"""

import os
import sys
import subprocess
import time

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("=" * 60)
    print("  DOOMSPHERE - ADVANCED DEPTH SYSTEM")
    print("=" * 60)
    print("")

def check_file(filename):
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found!")
        return False
    return True

def run_script(script_name, args=[]):
    if not check_file(script_name):
        input("\nPress Enter to continue...")
        return

    cmd = [sys.executable, script_name] + args
    print(f"\nLaunching {script_name}...")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error running script: {e}")
    
    print("\nProcess finished.")
    input("Press Enter to return to menu...")

def main():
    while True:
        clear_screen()
        print_header()
        print("Select Mode:")
        print("  1. Stereo Depth View (Dual Camera)")
        print("  2. MiDaS AI Depth (Single Camera)")
        print("  3. Stereo Calibration Tool")
        print("  4. Exit")
        print("")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            run_script("depth_camera.py")
        elif choice == '2':
            run_script("midas_depth.py")
        elif choice == '3':
            run_script("calibrate_stereo.py")
        elif choice == '4':
            print("\nExiting DoomSphere. Goodbye!")
            break
        else:
            print("\nInvalid choice!")
            time.sleep(1)

if __name__ == "__main__":
    main()
