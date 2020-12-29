import numpy as np

import skimage.io as skio
import matplotlib.pyplot as plt

def plot_frame_shift(ax, im, query_inliers, db_inliers):
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
    loc = np.array((0., 0., 0.))
    path_x = [0.]
    path_z = [0.]
    for R, T in pose_history[start_pose:]:
        shift = R.T.dot(T)
        loc += shift
        path_x.append(loc[1])
        path_z.append(loc[2])

    ax.scatter(P[1,:] - loc[1], P[2,:] - loc[2], marker='o', facecolors='none', edgecolors='b')
    ax.plot(path_x, path_z, "rx")
    ax.set_ylim(ymin=-20, ymax=20)
    ax.set_xlim(xmin=-20, xmax=20)
    # TODO: this needs to be adjusted for rotation, as we are moving to the side, everything is rotated by 90 degs

def plot_camera_pose(ax, pose_history):
    ax.clear()
    loc = np.array((0., 0., 0.))
    path_x = [0.]
    path_z = [0.]

    for R, T in pose_history:
        shift = R.T.dot(T)
        loc += shift
        path_x.append(loc[1])
        path_z.append(loc[2])

    ax.plot(path_x, path_z)
    return loc
