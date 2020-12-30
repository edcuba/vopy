import cv2
import numpy as np

def get_essential_matrix(points0, points1, K):
    E, mask = cv2.findEssentialMat(points1, points0, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    return E, mask[:,0].astype(bool)

def get_pose(E, points0, points1, K):
    _, R, T, _ = cv2.recoverPose(E, points1, points0, K)
    return R, T

def triangulate(points0, points1, R, T, K):
    M0 = K.dot(np.eye(3, 4))

    RT = np.hstack((R, T))
    M1 = K.dot(RT)

    p0 = points0.T.astype(np.float64)
    p1 = points1.T.astype(np.float64)

    triangulated = cv2.triangulatePoints(M0, M1, p0, p1)

    return triangulated[:3,:].T
