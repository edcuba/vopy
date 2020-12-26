# continuous operation of the pipeline

# a good resource for python: https://scikit-image.org/docs/dev/auto_examples/transform/plot_fundamental_matrix.html

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

from vopy.utils import load_dataset
from vopy.keypoints import harris, select_keypoints, describe_keypoints, match_descriptors, transform_coords
from vopy.pose import get_essential_matrix, decompose_essential_matrix, disambiguate_poses, triangulate
from vopy.plotting import plot_frame_shift, plot_landmarks, plot_camera_pose

DATASET_PATH = "data/parking"
LAST_FRAME = 598

RADIUS = 9
RATIO = 1
RANSAC_TRIALS = 1024
LAMBDA = 4

# Parameters to be tuned
#   descriptor radius = patch size
#   number of keypoints = how many patches
#   image ratio = downsampling the image for performance
#   match lambda = match sensitivity

def process_image(im, radius=RADIUS, keypoints=300):
    harris_scores = harris(im)
    keypoints = select_keypoints(harris_scores, num_keypoints=keypoints)
    descriptors = describe_keypoints(im, keypoints, desc_radius=radius)

    return harris_scores, keypoints, descriptors

get_image, ground_truth, K = load_dataset(DATASET_PATH, image_ratio=RATIO)

pose_history = []

fig = plt.figure(figsize=(14, 7))
ax_image = fig.add_subplot(2, 2, 1)
ax_full = fig.add_subplot(2, 2, 3)
ax_last20 = fig.add_subplot(1, 2, 2)

# initialize the world frame
im_prev = get_image(0)
h_prev, k_prev, d_prev = process_image(im_prev)
origin_prev = np.array((0, 0, 0))

# use 5th frame to triangulate the initial point cloud
harris_scores, keypoints, descriptors = process_image(get_image(5))
dists = match_descriptors(descriptors, d_prev, match_lambda=LAMBDA)
matched_query, matched_db = keypoints[dists[:,0]], k_prev[dists[:,1]]
E, query_inliers, db_inliers = get_essential_matrix(matched_query, matched_db, max_trials=RANSAC_TRIALS)
p0, p1 = transform_coords(db_inliers, query_inliers)
rots, u3 = decompose_essential_matrix(E)
R, T, PC1 = disambiguate_poses(rots, u3, p0, p1, K)

pose_history.append((R, T))
plot_camera_pose(ax_full, pose_history)

for i in range(1, LAST_FRAME+1):

    matched = True

    im = get_image(i)
    harris_scores, keypoints, descriptors = process_image(im)

    # match descriptors between images
    dists = match_descriptors(descriptors, d_prev, match_lambda=LAMBDA)

    # matches in the query and db image
    matched_query, matched_db = keypoints[dists[:,0]], k_prev[dists[:,1]]

    # recover essential matrix
    try:
        E, query_inliers, db_inliers = get_essential_matrix(matched_query, matched_db, max_trials=RANSAC_TRIALS)
    except ValueError:
        print(f"Frame {i} failed to localize")
        matched = False

    if matched:
        plot_frame_shift(ax_image, im, query_inliers, db_inliers)
        print(f"Frame {i} matched {matched_query.shape[0]}/{keypoints.shape[0]} inliers {query_inliers.shape[0]}")

        p0, p1 = transform_coords(db_inliers, query_inliers)

        # get essential matrix
        rots, u3 = decompose_essential_matrix(E)

        R, T, PC1 = disambiguate_poses(rots, u3, p0, p1, K)
        pose_history.append((R, T))

        plot_camera_pose(ax_full, pose_history)
        plot_landmarks(ax_last20, PC1)
    else:
        im_prev = im
        h_prev, k_prev, d_prev = harris_scores, keypoints, descriptors

    plt.pause(1e-6)

# TODO: need unit tests
#  - select first 2 frames and test against exercise 6 - can load arbitrary pictures
#  - once that works, only the continuous operation is missing
