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
    points_depth = depth_img[points[0], points[1]]
    points_depth = np.expand_dims(points_depth, axis=0)
    projected_points = np.append(points, points_depth, axis=0)

    K = np.array([fx, 0., cx,
                  0., fy, cy,
                  0., 0., 1])

    return (ids, projected_points)