import time

import cv2
import numpy as np
from typing import Tuple, List
from scipy.signal import convolve2d


def gauss_kernel(size):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    g = np.exp(-(x ** 2 / float(size) + y ** 2 / float(size)))
    return g / g.sum()


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
    start = time.time()
    i_x = cv2.Sobel(src=img, ddepth=-1, dx=1, dy=0)
    i_y = cv2.Sobel(src=img, ddepth=-1, dx=0, dy=1)
    gauss = gauss_kernel(3)
    A = convolve2d(i_x * i_x, gauss, mode="same") + 1e-15
    B = convolve2d(i_y * i_y, gauss, mode="same") + 1e-15
    C = convolve2d(i_x * i_y, gauss, mode="same") + 1e-15

    det = A * B - C * C
    trace = A + B
    R = det / (0.06 * trace)
    matches = R >= threshold
    print(np.count_nonzero(matches))
    indicies = np.nonzero(matches)
    res = R[matches]
    return points


def harris_corners_test(img: np.ndarray, threshold=1.0, blur_sigma=2.0) -> List[Tuple[float, np.ndarray]]:
    """
    Return the harris corners detected in the image.
    :param img: The grayscale image.
    :param threshold: The harris respnse function threshold.
    :param blur_sigma: Sigma value for the image bluring.
    :return: A sorted list of tuples containing response value and image position.
    The list is sorted from largest to smallest response value.
    """
    points = []
    start = time.time()
    i_x = cv2.Sobel(src=img, ddepth=-1, dx=1, dy=0)
    i_y = cv2.Sobel(src=img, ddepth=-1, dx=0, dy=1)
    gauss = gauss_kernel(3)
    A = convolve2d(i_x * i_x, gauss, mode="same") + 1e-15
    B = convolve2d(i_y * i_y, gauss, mode="same") + 1e-15
    C = convolve2d(i_x * i_y, gauss, mode="same") + 1e-15

    det = A * B - C * C
    trace = A + B
    R = det / 0.06*(trace**2)
    matches = R >= threshold
    img[matches] = 1
    return img
