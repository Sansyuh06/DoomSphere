import cv2
import numpy as np

def create_chessboard(rows, cols, square_size_px):
    # rows, cols are number of *squares*
    # For 7x7 inner corners, we need 8x8 squares
    width = cols * square_size_px
    height = rows * square_size_px
    image = np.zeros((height, width), dtype=np.uint8)
    
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 1:
                image[r*square_size_px : (r+1)*square_size_px,
                      c*square_size_px : (c+1)*square_size_px] = 255
    return image

# Standard chessboard is 8x8 squares -> 7x7 inner corners
board = create_chessboard(8, 8, 100)
# Add a white border
board = cv2.copyMakeBorder(board, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)

cv2.imwrite("digital_chessboard.png", board)
print("Created digital_chessboard.png. Open this file on your screen!")
