import cv2

def main():
    print("Testing cameras to identify indices...")
    # Test usually indices 0, 1, 2, 3
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera {i} is available. Opening preview... (Press 'q' to next)")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow(f"Camera {i}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            print(f"Camera {i} is not available.")

if __name__ == "__main__":
    main()
