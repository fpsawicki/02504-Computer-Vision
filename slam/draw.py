import cv2
import matplotlib.pyplot as plt
import numpy as np


def drawlines(img1, img2, lines, pts1, pts2):
    ''' 
        img1: image on which we draw the epilines for the points in img2
        lines: corresponding epilines 
    '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1[0]), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2[0]), 5, color, -1)
    return img1,img2


def epilines(img1, img2, pts1, pts2, F, size=(18.5, 10.5)):
    """
        img1: nparray, camera_1 image
        img2: nparray, camera_2 image
        pts1: nparray, camera_1 correspondnig feature points
        pts2: nparray, camera_2 correspondnig feature points
        F:    nparray, fundamental matrix
        size: tuple of ints, size of pyplot

        returns: draws pyplot of 2 images with corresponding epilines on feature points
    """
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1,3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F)
    lines2 = lines2.reshape(-1,3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img5)
    ax[1].imshow(img3)
    fig.set_size_inches(size[0], size[1], forward=True)
    plt.show()


def feature_matches(img1, img2, kp1, kp2, good, size=(18.5, 10.5)):
    """
        img1: nparray, camera_1 image
        img2: nparray, camera_2 image
        kp1: nparray, camera_1 correspondnig feature points
        kp2: nparray, camera_2 correspondnig feature points
        good: float, points chosen as good features

        returns: draws pyplot of 2 images with lines matching corresponding feature points
    """
    img3 = cv2.drawMatchesKnn(
        img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img3)
    fig.set_size_inches(size[0], size[1], forward=True)
    plt.show()