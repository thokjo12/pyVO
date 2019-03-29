import time

import cv2
import numpy as np
from operator import itemgetter
from typing import Tuple, List
import scipy.ndimage.filters as filter


def harris_corners(img: np.ndarray, threshold=1.0, blur_sigma=2.0) -> List[Tuple[float, np.ndarray]]:
    """
    Return the harris corners detected in the image.
    :param img: The grayscale image.
    :param threshold: The harris respnse function threshold.
    :param blur_sigma: Sigma value for the image bluring.
    :return: A sorted list of tuples containing response value and image position.
    The list is sorted from largest to smallest response value.
    """
    points = []
    # find i_x  and i_y from a sobel filter

    # instead of sobel look into Scharr as i might produce something more accurate
    i_x = cv2.Sobel(src=img, ddepth=-1, dx=1, dy=0)
    i_y = cv2.Sobel(src=img, ddepth=-1, dx=0, dy=1)
    i_xx = filter.gaussian_filter(i_x * i_x, blur_sigma)
    i_xy = filter.gaussian_filter(i_x * i_y, blur_sigma)
    i_yy = filter.gaussian_filter(i_y * i_y, blur_sigma)

    m = np.empty((2, 2))
    kernel_size = 11
    final_y = img.shape[0] - kernel_size
    final_x = img.shape[1] - kernel_size

    for (y, x), v in np.ndenumerate(img):
        if x <= final_x or y <= final_y:
            A = i_xx[y:y + kernel_size, x:x + kernel_size].sum()
            B = i_yy[y:y + kernel_size, x:x + kernel_size].sum()
            C = i_xy[y:y + kernel_size, x:x + kernel_size].sum()

            m[0, 0] = A
            m[0, 1] = B
            m[1, 0] = B
            m[1, 1] = C
            R = np.linalg.det(m) - 0.06 * (np.trace(m) * np.trace(m))
            if R >= threshold:
                points.append((R, np.array([y, x])))

    points.sort(key=lambda R: R[0],reverse=True)

    return points