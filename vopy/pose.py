import numpy as np

from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform


def get_essential_matrix(keypoints_query, keypoints_db, max_trials=2000):
    matches = (keypoints_db, keypoints_query)
    model, inliers = ransac(
        matches,
        EssentialMatrixTransform,
        min_samples=8,
        max_trials=max_trials,
        residual_threshold=0.5
    )
    query_inliers, db_inliers = keypoints_query[inliers], keypoints_db[inliers]
    return model.params, query_inliers, db_inliers


def decompose_essential_matrix(E):
    U, _, V = np.linalg.svd(E, full_matrices=True)
    u3 = U[:,-1]
    W = np.array(((0, -1, 0), (1, 0, 0), (0, 0, 1)))

    R1 = U * W * V.T
    R2 = U * W.T * V.T

    if np.linalg.det(R1) < 0:
        R1 = -R1

    if np.linalg.det(R2) < 0:
        R2 = -R2

    if np.linalg.norm(u3) != 0:
        u3 = u3 / np.linalg.norm(u3)

    R = np.array((R1, R2))
    return R, u3

def cross_matrix(x):
    return np.array(
        ((0., -x[2], x[1]),
        (x[2], 0., -x[0]),
        (-x[1], x[0], 0.)),
        dtype=np.float64
    )

def triangulate(points0, points1, M1, M2):
    num_points = points0.shape[1]
    P = np.zeros((4, num_points))
    for i in range(num_points):
        p0 = points0[:, i]
        p1 = points1[:, i]
        c0 = cross_matrix(p0)
        c1 = cross_matrix(p1)
        A1 = c0.dot(M1)
        A2 = c1.dot(M2)
        A = np.vstack((A1, A2))
        _, _, V = np.linalg.svd(A)
        P[:,i] = V[:,-1]
    return P / np.tile(P[-1,:], (4, 1))

def disambiguate_poses(rots, u3, points0, points1, K):
    M1 = K.dot(np.eye(3, 4))
    R = rots[0]
    T = u3

    best_pose_points = 0
    PC1 = np.array([])

    for rot in rots:
        for sign in range(2):
            trans = u3 * (-1)**sign
            RT = np.hstack((rot, trans[:, np.newaxis]))
            M2 = K.dot(RT)
            P_C1 = triangulate(points0, points1, M1, M2)
            P_C2 = RT.dot(P_C1)
            points_visible_1 = sum(P_C1[2,:] > 0)
            points_visible_2 = sum(P_C2[2,:] > 0)
            points_visible = points_visible_1 + points_visible_2

            if points_visible > best_pose_points:
                R = rot
                T = trans
                best_pose_points = points_visible
                PC1 = P_C1

    return R, T, PC1
