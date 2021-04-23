import numpy as np
import cv2


def choose_points(src_pts, dst_pts, choices):
    if choices > src_pts.shape[0]:
        raise Exception(f'Invalid number of choices, max: {src_pts.shape[0]}')
    
    corrs = []
    choices = np.random.choice(src_pts.shape[0], size=choices, replace=False)
    for i in choices:
        corrs.append((src_pts[i], dst_pts[i]))
        # normalize points ?
    return np.array(corrs)

    
def find_projection_matrix(camera_matrix, src_pts, dst_pts, rot1, rot2, trans, choices=10):
    """ Tries to find unique solution for projection matrix

        camera_matrix: numpy array of calibrated camera (we assume that both cameras have the same matrix)
        src_pts: camera_1 feature points
        dst_pts: camera_2 feature points
        rot1: rotation_1 from essential matrix decomposition
        rot2: rotation_2 from essential matrix decomposition
        trans: translation from essential matrix decomposition
        choices: how many random source/destination points to use for finding projection matrix 

        returns: dictionary with projection matrices for 2 cameras, translation vector and rotation_translation for 2nd camera
    """
    # creates projection matrix for the reference (first) camera
    rt_mat_orig = np.hstack((np.identity(3), np.zeros(3)[np.newaxis].T))
    projection_mat_orig = np.dot(camera_matrix, rt_mat_orig)
    
    solutions = []
    points = choose_points(src_pts, dst_pts, choices)
    combinations = [(rot1, trans), (rot1, -trans), (rot2, trans), (rot2, -trans)]
    for rot, t in combinations:
        # creates projection matrix for the second camera
        rt_mat_2nd = np.hstack((rot, t))
        projection_mat_2nd = np.dot(camera_matrix, rt_mat_2nd)
        
        pts_3d = cv2.triangulatePoints(
            projection_mat_orig, projection_mat_2nd, points[:, 0], points[:, 1]
        )
        uhomo_pts_3d = np.array([pts_3d[0]/pts_3d[3], pts_3d[1]/pts_3d[3], pts_3d[2]/pts_3d[3]])
        if np.any(uhomo_pts_3d[2, :] < 0):
            continue  # invalid solution, point is behind the camera

        solutions.append({
            'pro_mat_1st': projection_mat_orig,
            'pro_mat_2nd': projection_mat_2nd,
            't_vec': t,
            'rt_mat': rt_mat_2nd
        })
    if len(solutions) > 1:
        choices += 1
        if choices > src_pts.shape[0]:
            raise Exception('Couldnt find unique solution to point triangulation')
        return find_projection_matrix(
            camera_matrix, src_pts, dst_pts, rot1, rot2, trans, choices=choices
        )
    if not solutions:
        raise Exception('Couldnt find any solution to point triangulation')
    return solutions[0]


def calc_fundamental_matrix(camera_matrix, essential_matrix):
    pinv_camera_t = np.linalg.inv(camera_matrix.T)
    pinv_camera = np.linalg.inv(camera_matrix)
    x = np.dot(pinv_camera_t, essential_matrix)
    F = np.dot(x, pinv_camera)  # C^-T * E * C^-1
    F = F / F[-1, -1]
    return F