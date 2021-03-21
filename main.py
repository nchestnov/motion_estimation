import numpy as np
import cv2

from src.lucas_kanade import optical_flow_pyr
from src.io import read_frames
from src.report import create_timing_graph, save_results


def estimate_motion(frames):
    result_translations = []
    result_time = []

    frame1 = frames[0]
    for frame2 in frames[1:]:
        translation, time_took = optical_flow_pyr(frame1, frame2)
        result_translations.append(translation)
        result_time.append(time_took)
    return np.array(result_translations), np.array(result_time)


if __name__ == '__main__':
    input_frames = read_frames(folder='input')

    result_translations, result_times = estimate_motion(input_frames)
    rounded_results = np.round(result_translations).astype(np.int32)

    save_results(rounded_results, 'results.txt')
    create_timing_graph(result_times)
