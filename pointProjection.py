import numpy as np
import cv2

from typing import Tuple

fx = 520.9
fy = 521.0
cx = 325.1
cy = 249.7


def project_points(ids: np.ndarray, points: np.ndarray, depth_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects the 2D points to 3D using the depth image and the camera instrinsic parameters.
    :param ids: A N vector point ids.
    :param points: A 2xN matrix of 2D points
    :param depth_img: The depth image. Divide pixel value by 5000 to get depth in meters.
    :return: A tuple containing a N vector and a 3xN vector of all the points that where successfully projected.
    """
    points = points.astype('int')
    Z = depth_img[points[1], points[0]] / 5000
    X = (points[0] - cx) * Z / fx
    Y = (points[1] - cy) * Z / fy
    projected_points = np.block([[X], [Y], [Z]])

    return (ids, projected_points)
