import time

import cv2
import numpy as np
from typing import Tuple, List

# Kernel size of the NMS filter
kernel = np.ones((27,27))


def harris_corners(img: np.ndarray, threshold=1.0, blur_sigma=2.0) -> List[Tuple[float, np.ndarray]]:
    """
    Return the harris corners detected in the image.
    :param img: The grayscale image.
    :param threshold: The harris response function threshold.
    :param blur_sigma: Sigma value for the image bluring.
    :return: A sorted list of tuples containing response value and image position.
    The list is sorted from largest to smallest response value.
    """
    # Get the directional derivatives of the image
    i_x = cv2.Scharr(src=img, ddepth=-1, dx=1, dy=0)
    i_y = cv2.Scharr(src=img, ddepth=-1, dx=0, dy=1)

    # Convolve and blur the derivatives to create the [[A, B], [B, C]] matrix
    A = cv2.GaussianBlur(i_x * i_x, (3, 3), blur_sigma)
    C = cv2.GaussianBlur(i_y * i_y, (3, 3), blur_sigma)
    B = cv2.GaussianBlur(i_x * i_y, (3, 3), blur_sigma)

    # Look into eigen values and find the best response values
    det = A * C - B * B
    trace = A + C
    R = det - 0.06 * trace * trace

    # Perform non max suppression
    dilated = cv2.dilate(R, kernel)
    matched = R == dilated
    R[~matched] = 0

    # Find applicable coordinates matching threshold
    indices = np.where(R >= threshold)
    coordinates = np.flip(np.stack(indices, axis=1),axis=1)

    # Extract points and sort
    points = list(zip(R[indices], coordinates))
    points.sort(key=lambda x: x[0], reverse=True)

    return points
