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

def plot_landmarks(ax, P):
    ax.clear()
    ax.scatter(P[0,:], P[2,:])

def plot_camera_pose(ax, pose_history):
    ax.clear()
    loc = np.array((0., 0., 0.))
    path_x = [0.]
    path_z = [0.]

    for R, T in pose_history:
        shift = -R.T.dot(T)
        loc += shift
        path_x.append(loc[0])
        path_z.append(loc[2])

    ax.plot(path_x, path_z)
