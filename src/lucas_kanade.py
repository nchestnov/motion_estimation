import numpy as np
import cv2

from scipy.ndimage.filters import gaussian_filter

import time


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        time_took = (time2-time1)*1000.0

        return ret, time_took
    return wrap


def optical_flow(img1, img2, points, window=3, threshold=0.04, weighted=True):
    img1 = img1.astype(np.float64) / 255.0
    img2 = img2.astype(np.float64) / 255.0

    img_dx = cv2.Scharr(img1, ddepth=-1, dx=1, dy=0) / 64.0
    img_dy = cv2.Scharr(img1, ddepth=-1, dx=0, dy=1) / 64.0
    img_dt = cv2.GaussianBlur(img2 - img1, ksize=(3, 3), sigmaX=0, sigmaY=0)

    translations = []
    translations_clear = []

    for b, a in points:
        a, b = int(a), int(b)
        Ix = img_dx[a - window:a + window + 1, b - window:b + window + 1]
        Iy = img_dy[a - window:a + window + 1, b - window:b + window + 1]
        It = img_dt[a - window:a + window + 1, b - window:b + window + 1]

        if weighted:
            a = np.zeros((2 * window + 1, 2 * window + 1))
            a[window, window] = 1
            weights = gaussian_filter(a, sigma=2)
        else:
            weights = np.ones((2 * window + 1, 2 * window + 1))

        A = np.float32([
            [np.sum(weights*Ix * Ix), np.sum(weights*Ix * Iy)],
            [np.sum(weights*Iy * Ix), np.sum(weights*Iy * Iy)]
        ])
        B = np.float32([
            -np.sum(weights*Ix * It),
            -np.sum(weights*Iy * It)
        ])

        translation = np.linalg.pinv(A) @ B

        smallest_eigval = np.min(np.linalg.eigvals(A))

        translations.append(translation)
        if threshold < smallest_eigval:
            translations_clear.append(translation)

    translations = np.array(translations)
    translations_clear = np.array(translations_clear)

    translation = np.median(translations, axis=0)
    if translations_clear.shape[0] > 0:
        translation_clear = np.median(translations_clear, axis=0)
        return translation_clear
    else:
        return translation


@timing
def optical_flow_pyr(img1, img2, pyr_level=6, window=2, num_points=20, threshold=0.02):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    img1_pyramid = [img1]
    img2_pyramid = [img2]

    points = cv2.goodFeaturesToTrack(img1, maxCorners=num_points, qualityLevel=0.7, minDistance=20)[:, 0, :]
    points = points.astype(np.int32)
    points_pyramid = [points]

    pyramid_shapes = [np.array(img1.shape)[::-1]]
    for _ in range(pyr_level):
        pyramid_shapes.append(pyramid_shapes[-1] // 2)
        img1_pyramid.append(cv2.pyrDown(img1_pyramid[-1], pyramid_shapes[-1]) / 255.0)
        img2_pyramid.append(cv2.pyrDown(img2_pyramid[-1], pyramid_shapes[-1]) / 255.0)
        points_pyramid.append(points_pyramid[-1] // 2)

    translation = np.zeros(2)
    for i in reversed(range(pyr_level)):
        translation *= 2
        img1_loc = img1_pyramid[i]
        img2_loc = img2_pyramid[i]
        points_loc = points_pyramid[i]  # col, row

        img1_loc = cv2.warpAffine(
            img1_loc,
            np.float32([
                [1.0, 0.0, translation[0]],
                [0.0, 1.0, translation[1]]]),
            img1_loc.shape[::-1]
        )

        translation += optical_flow(img1_loc, img2_loc, points_loc, window=window, threshold=threshold)

    return translation
