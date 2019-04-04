import time

import cv2
import numpy as np
from typing import Tuple, List
from scipy.signal import convolve2d, windows


# TODO: replace with the scipy signal function.
def gauss_kernel(size):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    g = np.exp(-(x ** 2 / float(size) + y ** 2 / float(size)))
    return g / g.sum()

def gauss_kernel2(size, sigma):
    gauss_filter = windows.gaussian(size, sigma,sym=True)
    return np.outer(gauss_filter, gauss_filter)


def harris_corners(img: np.ndarray, threshold=1.0, blur_sigma=2.0) -> List[Tuple[float, np.ndarray]]:
    """
    Return the harris corners detected in the image.
    :param img: The grayscale image.
    :param threshold: The harris response function threshold.
    :param blur_sigma: Sigma value for the image bluring.
    :return: A sorted list of tuples containing response value and image position.
    The list is sorted from largest to smallest response value.
    """
    i_x = cv2.Scharr(src=img, ddepth=-1, dx=1, dy=0)
    i_y = cv2.Scharr(src=img, ddepth=-1, dx=0, dy=1)

    window = gauss_kernel(3)
    A = convolve2d(i_x * i_x, window, mode="same")
    C = convolve2d(i_y * i_y, window, mode="same")
    B = convolve2d(i_x * i_y, window, mode="same")

    det = A * C - B * B
    trace = A + C
    R = det - 0.06 * trace * trace

    # TODO: introduce NMS

    indices = np.where(R >= threshold)
    coordinates = np.stack(indices, axis=1)
    points = list(zip(R[indices],coordinates))

    points.sort(key=lambda x:x[0],reverse=True)
    return points


# TODO: replace with the scipy signal function.
def gauss_kernel(size):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    g = np.exp(-(x ** 2 / float(size) + y ** 2 / float(size)))
    return g / g.sum()

def gauss_kernel2(size, sigma):
    gauss_filter = windows.gaussian(size, sigma,sym=True)
    return np.outer(gauss_filter, gauss_filter)