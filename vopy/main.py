# continuous operation of the pipeline

# a good resource for python: https://scikit-image.org/docs/dev/auto_examples/transform/plot_fundamental_matrix.html

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

from vopy.utils import load_dataset
from vopy.keypoints import harris, select_keypoints, describe_keypoints, match_descriptors
from vopy.pose import get_essential_matrix, decompose_essential_matrix, disambiguate_poses, triangulate
from vopy.plotting import plot_frame_shift, plot_point_cloud, plot_trajectory

DATASET_PATH = "data/parking"
LAST_FRAME = 598

# Perf optimization
if False:
    RADIUS = 4
    RATIO = 0.5
    RANSAC_TRIALS = 500
    LAMBDA = 10
else:
    RADIUS = 9
    RATIO = 1
    RANSAC_TRIALS = 1024
    LAMBDA = 4

# Parameters to be tuned
#   descriptor radius = patch size
#   number of keypoints = how many patches
#   image ratio = downsampling the image for performance
#   match lambda = match sensitivity

def process_image(im, radius=RADIUS, keypoints=200):
    harris_scores = harris(im)
    keypoints = select_keypoints(harris_scores, num_keypoints=keypoints)
    descriptors = describe_keypoints(im, keypoints, desc_radius=radius)

    return harris_scores, keypoints, descriptors

get_image, ground_truth, K = load_dataset(DATASET_PATH, image_ratio=RATIO)


# initialize the world frame
im_prev = get_image(0)
h_prev, k_prev, d_prev = process_image(im_prev)

fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax3 = fig.add_subplot(2, 2, 4)


camera_pos = np.array((0., 0., 0.))

ax2.set_xlim([-1, 100])
ax2.set_ylim([-1, 100])
ax2.set_zlim([-1, 100])

ax3.set_xlim([-1, 20])
ax3.set_ylim([-1, 5])
ax3.plot(camera_pos[0], camera_pos[2], "o")


for i in range(1, LAST_FRAME+1):
    im = get_image(i)
    harris_scores, keypoints, descriptors = process_image(im)

    # match descriptors between images
    dists = match_descriptors(descriptors, d_prev, match_lambda=LAMBDA)

    # matches in the query and db image
    matched_query, matched_db = keypoints[dists[:,0]], k_prev[dists[:,1]]

    # recover essential matrix
    model, query_inliers, db_inliers = get_essential_matrix(matched_query, matched_db, max_trials=RANSAC_TRIALS)

    plot_frame_shift(ax1, im, query_inliers, db_inliers)
    print(f"Frame {i} matched {matched_query.shape[0]}/{keypoints.shape[0]} inliers {query_inliers.shape[0]}")

    n = db_inliers.shape[0]
    p0 = np.hstack((db_inliers, np.ones(n)[:, np.newaxis]))
    p1 = np.hstack((query_inliers, np.ones(n)[:, np.newaxis]))

    # get essential matrix
    E = model.params
    rots, u3 = decompose_essential_matrix(E)

    R, T, PC1 = disambiguate_poses(rots, u3, p0, p1, K)
    RT = np.hstack((R, T[:, np.newaxis]))


    camera_pos += T
    ax3.plot(camera_pos[0], camera_pos[2], "o")
    plot_trajectory(ax3, camera_pos, PC1)
    plot_point_cloud(ax2, camera_pos, PC1)

    im_prev = im
    h_prev, k_prev, d_prev = harris_scores, keypoints, descriptors
    plt.pause(1e-6)
