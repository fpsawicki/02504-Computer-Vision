import cv2
import numpy as np


def sift_features(img1, img2, good_ratio=0.8, matcher='bf'):
    """
        img1: nparray, camera_1 image
        img2: nparray, camera_2 image
        good_ratio: float, defines how good feature matches to return
        matcher: str (bf or flann), selects matching algorithm

        returns: feature points of camera_1 and camera_2, and list of good matches
    """
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    if matcher == 'flann':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good = []

        for i, (m, n) in enumerate(matches):
            if m.distance < good_ratio * n.distance:
                good.append([m])
        return kp1, kp2, good
    
    if matcher == 'bf':
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []

        for m, n in matches:
            if m.distance < good_ratio * n.distance:
                good.append([m])
        return kp1, kp2, good


def select_good_features(kp1, kp2, good):
    """
        kp1: list of features on camera_1
        kp2: list of features on camera_2
        good: list of good feature matches

        returns: filtered features of both cameras with good matches
    """
    pts_im1 = [kp1[m[0].queryIdx].pt for m in good]
    pts_im1 = np.array(pts_im1, dtype=int)
    pts_im2 = [kp2[m[0].trainIdx].pt for m in good]
    pts_im2 = np.array(pts_im2, dtype=int)
    return pts_im1, pts_im2