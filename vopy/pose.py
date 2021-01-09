import cv2
import numpy as np

def get_essential_matrix(points0, points1, K):
    E, mask = cv2.findEssentialMat(points1, points0, K, method=cv2.RANSAC, prob=0.999, threshold=2)
    return E, mask[:,0].astype(bool)

def get_pose(E, points0, points1, K):
    _, R, T, _ = cv2.recoverPose(E, points1, points0, K)
    return R, T

def triangulate(points0, points1, R, T, K):
    M0 = K.dot(np.eye(3, 4))

    RT = np.hstack((R, T))
    M1 = K.dot(RT)

    points0f = points0.T.astype(np.float32)
    points1f = points1.T.astype(np.float32)
    points4d_hom = cv2.triangulatePoints(M0, M1, points0f, points1f)

    # dehomogenize the points
    points4d_hom_good = points4d_hom[:,points4d_hom[3] != 0]
    points4d = points4d_hom_good / points4d_hom_good[3]

    points3d = points4d[:3,:].T

    # reject points behind the camera frame
    points3d_good = points3d[points3d[:,2] < 0,:]

    return points3d_good
