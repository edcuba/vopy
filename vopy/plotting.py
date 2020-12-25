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

def plot_point_cloud(ax, camera_pos, P):
    ax.scatter(P[0] + camera_pos[0], P[1] + camera_pos[1], P[2] + camera_pos[2])

def plot_trajectory(ax, camera_pos, P):
    x_pose = P[0] + camera_pos[0]
    z_pose = P[2] + camera_pos[2]
    ax.plot(x_pose, z_pose, 'rx')
