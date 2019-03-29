import cv2
import numpy as np
from operator import itemgetter
from typing import Tuple, List


def harris_corners(img: np.ndarray, threshold=1.0, blur_sigma=2.0) -> List[Tuple[float, np.ndarray]]:
    """
    Return the harris corners detected in the image.
    :param img: The grayscale image.
    :param threshold: The harris respnse function threshold.
    :param blur_sigma: Sigma value for the image bluring.
    :return: A sorted list of tuples containing response value and image position.
    The list is sorted from largest to smallest response value.
    """
    # find i_x  and i_y from a sobel filter
    i_x = None
    i_y = None

    # instead of sobel look into Scharr as i might produce something more accurate
    i_x = cv2.Sobel(src=img, ddepth=-1, dx=1, dy=0)
    i_y = cv2.Sobel(src=img, ddepth=-1, dx=0, dy=1)

    win_func = None
    M = np.block([[i_x*i_x,i_x*i_y],
                           [i_x*i_y,i_y*i_y]])

    # m = w(x,y) dot [ix * ix,ix * iy
    # ;ix* iy,iy * iy]

    raise NotImplementedError
