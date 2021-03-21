import numpy as np
import cv2


def read_frames(folder='input'):
    frames = []
    for i in range(1, 13):
        frames.append(cv2.imread(f'{folder}/{i:02d}.tif', 0))
    return np.array(frames)
