import numpy as np
import matplotlib.pyplot as plt

def plot_frame_shift(ax, im, db_inliers, query_inliers):
    ax.clear()
    ax.imshow(im, cmap='gray')
    ax.plot(db_inliers[:,1], db_inliers[:,0], "rx")

    x_from = db_inliers[:,1]
    x_to = query_inliers[:,1]

    y_from = db_inliers[:,0]
    y_to = query_inliers[:,0]

    x = np.vstack([x_from, x_to])
    y = np.vstack([y_from, y_to])

    # show their paths in the new image
    ax.plot(x, y, "g-")

def plot_landmarks(ax, P, pose_history):
    ax.clear()

    start_pose = max(0, len(pose_history) - 20)

    cR = np.eye(3, 3)
    cT = np.ones((3, 1))

    scale = 1

    path_y = [cT[1]]
    path_z = [cT[2]]

    for R, T in pose_history[start_pose:]:
        cT = cT + scale * cR.dot(T)
        cR = R.dot(cR)
        path_y.append(cT[1])
        path_z.append(cT[2])

    points = - 20 * P.T

    ax.scatter(points[1,:], points[2,:], marker='o', facecolors='none', edgecolors='b')
    ax.plot(path_y - cT[1], path_z - cT[2], "rx")
    ax.set_ylim(ymin=-40, ymax=40)
    ax.set_xlim(xmin=-40, xmax=40)

    # TODO: Need to dehomogenize the points

def plot_camera_pose(ax, pose_history):
    ax.clear()

    cR = np.eye(3, 3)
    cT = np.ones((3, 1))
    scale = 1

    path_y = [cT[1]]
    path_z = [cT[2]]

    for R, T in pose_history:
        cT = cT + scale * cR.dot(T)
        cR = R.dot(cR)
        path_y.append(cT[1])
        path_z.append(cT[2])

    ax.plot(path_y, path_z)
