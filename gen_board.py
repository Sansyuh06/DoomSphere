import cv2
import numpy as np

rows, cols = 8, 8
sq = 80

h = rows * sq
w = cols * sq
board = np.ones((h, w), dtype=np.uint8) * 255

for r in range(rows):
    for c in range(cols):
        if (r + c) % 2 == 1:
            board[r*sq:(r+1)*sq, c*sq:(c+1)*sq] = 0

border = 40
final = np.ones((h + border*2, w + border*2), dtype=np.uint8) * 255
final[border:border+h, border:border+w] = board

cv2.imwrite("chessboard.png", final)
print(f"Saved chessboard.png ({cols}x{rows} squares, {cols-1}x{rows-1} inner corners)")
print("Print this on A4 paper and use it for calibration!")
