import cv2
import numpy as np

import config


def find_corners(calibration_imgs, show_corners=False):
    """
        calibration_imgs: list of calibration images, can more than one for experimentation
        show_corners: bool, displays image with found corners

        returns: objectpoints, imagepoints and last grayscaled calibration image
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    n_rows, n_cols = config.CHECKBOARD[0], config.CHECKBOARD[1]
    objp = np.zeros((n_rows * n_cols, 3), np.float32)
    objp[:,:2] = np.mgrid[0:n_rows, 0:n_cols].T.reshape(-1,2)

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for gray in calibration_imgs:
        ret, corners = cv2.findChessboardCorners(gray, (n_rows, n_cols), None)
        if not ret:
            continue

        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        if show_corners:
            img = cv2.drawChessboardCorners(img, (n_rows, n_cols), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(0)
    
    if show_corners:
        cv2.destroyAllWindows()
    return objpoints, imgpoints, gray


def calibrate(calibration_imgs, show_corners=False):
    """
        calibration_imgs: list of calibration images, can more than one for experimentation
        show_corners: bool, displays image with found corners

        returns: camera matrix
    """
    objpoints, imgpoints, gray = find_corners(calibration_imgs, show_corners)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    if not ret:
        raise Exception("Camera calibration failed")
    return mtx
