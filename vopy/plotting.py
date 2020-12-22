import numpy as np

import skimage.io as skio
import matplotlib.pyplot as plt

def plot_frame_shift(ax, im, query_inliers, db_inliers):
    ax.clear()
    skio.imshow(im)
    plt.plot(db_inliers[:,1], db_inliers[:,0], "rx")

    x_from = db_inliers[:,1]
    x_to = query_inliers[:,1]

    y_from = db_inliers[:,0]
    y_to = query_inliers[:,0]

    x = np.vstack([x_from, x_to])
    y = np.vstack([y_from, y_to])

    # show their paths in the new image
    plt.plot(x, y, "g-")

    plt.pause(1e-6)
