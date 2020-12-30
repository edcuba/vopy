# continuous operation of the pipeline

# a good resource for python: https://scikit-image.org/docs/dev/auto_examples/transform/plot_fundamental_matrix.html

# Import libraries
import cv2
import numpy as np

import matplotlib.pyplot as plt

from vopy.utils import load_dataset
from vopy.keypoints import harris, select_keypoints, describe_keypoints, match_descriptors, transform_coords
from vopy.pose import get_essential_matrix, get_pose, triangulate
from vopy.plotting import plot_frame_shift, plot_landmarks, plot_camera_pose

kitti = False

if kitti:
    DATASET_PATH = "data/kitti/00"
    LAST_FRAME = 999
    START_FRAME = 50
else:
    DATASET_PATH = "data/parking"
    LAST_FRAME = 598
    START_FRAME = 0

RADIUS = 9
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

get_image, ground_truth, K = load_dataset(DATASET_PATH, kitti=kitti)

pose_history = []

fig = plt.figure(figsize=(14, 7))
ax_image = fig.add_subplot(2, 2, 1)

ax_full = fig.add_subplot(2, 2, 3)
ax_full.set_aspect('equal', 'datalim')

ax_last20 = fig.add_subplot(1, 2, 2)
ax_last20.set_aspect('equal', 'datalim')

# initialize the world frame
im_prev = get_image(START_FRAME)
h_prev, k_prev, d_prev = process_image(im_prev)
origin_prev = np.array((0, 0, 0))

# use 5th frame to triangulate the initial point cloud
harris_scores, keypoints, descriptors = process_image(get_image(START_FRAME+5))
dists = match_descriptors(descriptors, d_prev, match_lambda=LAMBDA)
matched_query, matched_db = keypoints[dists[:,0]], k_prev[dists[:,1]]
E, mask = get_essential_matrix(matched_db, matched_query, K)
p0_in, p1_in = matched_db[mask], matched_query[mask]

R, T = get_pose(E, p0_in, p1_in, K)
pose_history.append((R, R.dot(T)))

for i in range(START_FRAME, LAST_FRAME+1):
    print(f"Frame {i}")

    matched = True

    im = get_image(i)
    harris_scores, keypoints, descriptors = process_image(im)

    # match descriptors between images
    dists = match_descriptors(descriptors, d_prev, match_lambda=LAMBDA)

    # matches in the query and db image
    matched_query, matched_db = keypoints[dists[:,0]], k_prev[dists[:,1]]

    E, mask = get_essential_matrix(matched_db, matched_query, K)
    p0_in, p1_in = matched_db[mask], matched_query[mask]

    R, T = get_pose(E, p0_in, p1_in, K)

    pose_history.append((R, R.dot(T)))

    cloud = triangulate(p0_in, p1_in, R, T, K)
    plot_frame_shift(ax_image, im, p0_in, p1_in)

    plot_camera_pose(ax_full, pose_history)
    plot_landmarks(ax_last20, cloud, pose_history)

    if not matched or i % 3 == 0:
        im_prev = im
        h_prev, k_prev, d_prev = harris_scores, keypoints, descriptors

    plt.pause(1e-6)

# TODO: use ground truth
