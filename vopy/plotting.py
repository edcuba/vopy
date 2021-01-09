import numpy as np
import matplotlib.pyplot as plt

def plot_frame_shift(ax, im, db_inliers, query_inliers):
    ax.clear()
    ax.imshow(im, cmap='gray')
    ax.plot(db_inliers[:,1], db_inliers[:,0], "rx")
    ax.set_ylim(ymin=im.shape[0], ymax=0)
    ax.set_xlim(xmin=0, xmax=im.shape[1])

    x_from = db_inliers[:,1]
    x_to = query_inliers[:,1]

    y_from = db_inliers[:,0]
    y_to = query_inliers[:,0]

    x = np.vstack([x_from, x_to])
    y = np.vstack([y_from, y_to])

    # show their paths in the new image
    ax.plot(x, y, "g-")

def plot_landmarks(ax, P, T_history, R_total):
    ax.clear()
    ax.set_title("Landmarks and last 20 frames trajectory")

    start_pose = max(0, len(T_history) - 20)

    path_y = [T[1] for T in T_history[start_pose:]]
    path_z = [T[2] for T in T_history[start_pose:]]

    cT = T_history[-1]

    points = -1 * R_total.dot(P.T)

    ax.scatter(points[1,:], points[2,:], marker='o', facecolors='none', edgecolors='b')
    ax.plot(path_y - cT[1], path_z - cT[2], "rx")
    ax.set_ylim(ymin=-50, ymax=50)
    ax.set_xlim(xmin=-50, xmax=50)

def plot_camera_pose(ax, T_history):
    ax.clear()
    ax.set_title("Full trajectory")
    path_y = [T[1] for T in T_history]
    path_z = [T[2] for T in T_history]
    ax.plot(path_y, path_z)

def plot_matches(ax, match_history):
    ax.clear()

    ax.set_xlabel("Frame")
    ax.set_ylabel("Tracked landmarks")
    ax.set_title("# tracked over last 50 frames")

    start_pose = max(0, len(match_history) - 50)
    frame = [f for f, _ in match_history[start_pose:]]
    matches = [m for _, m in match_history[start_pose:]]

    ax.plot(frame, matches)
