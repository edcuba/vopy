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


kitti = True
lkt = False

if kitti:
    DATASET_PATH = "data/kitti/00"
    LAST_FRAME = 999
    START_FRAME = 50
else:
    DATASET_PATH = "data/parking"
    LAST_FRAME = 10 #598
    START_FRAME = 0

INIT_FRAME = START_FRAME + 2
RADIUS = 9
LAMBDA = 4
KEYPOINTS = 250

if lkt:
    RADIUS = 7

lkt_flow = {
    "winSize": (2 * RADIUS + 1, 2 * RADIUS + 1),
    "maxLevel": 2,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
}

lkt_features = {
    "maxCorners": KEYPOINTS,
    "qualityLevel": 0.5,
    "minDistance": RADIUS,
    "blockSize": RADIUS
}

# Parameters to be tuned
#   descriptor radius = patch size
#   number of keypoints = how many patches
#   image ratio = downsampling the image for performance
#   match lambda = match sensitivity

def process_image(im, keypoints=KEYPOINTS):
    harris_scores = harris(im)
    return select_keypoints(harris_scores, num_keypoints=keypoints)

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

if lkt:
    im_prev_t = im_prev.T # openCV is C++
    p_prev = cv2.goodFeaturesToTrack(im_prev_t, mask = None, **lkt_features)
else:
    k_prev = process_image(im_prev)
    d_prev = describe_keypoints(im_prev, k_prev, desc_radius=RADIUS)
    im = get_image(INIT_FRAME)
    keypoints = process_image(im)
    descriptors = describe_keypoints(im, keypoints, desc_radius=RADIUS)
    dists = match_descriptors(descriptors, d_prev, match_lambda=LAMBDA)
    matched_query, matched_db = keypoints[dists[:,0]], k_prev[dists[:,1]]
    E, mask = get_essential_matrix(matched_db, matched_query, K)
    p0_in, p1_in = matched_db[mask], matched_query[mask]
    R, T = get_pose(E, p0_in, p1_in, K)
    pose_history.append((R, R.dot(T)))

for i in range(INIT_FRAME + 1, LAST_FRAME + 1):
    im = get_image(i)

    if lkt:
        im_t = im.T # openCV is C++
        if p_prev.shape[0] < 50:
            p_prev = cv2.goodFeaturesToTrack(im_prev_t, mask = None, **lkt_features)
        p_query, st, _ = cv2.calcOpticalFlowPyrLK(im_prev_t, im_t, p_prev, None, **lkt_flow)
        matched_db = p_prev[st == 1]
        matched_query = p_query[st == 1]
        im_prev_t = im_t
    else:
        keypoints = process_image(im)
        descriptors = describe_keypoints(im, keypoints, desc_radius=RADIUS)

        # match descriptors between images
        dists = match_descriptors(descriptors, d_prev, match_lambda=LAMBDA)

        # matches in the query and db image
        matched_query, matched_db = keypoints[dists[:,0]], k_prev[dists[:,1]]
        k_prev, d_prev = keypoints, descriptors
        im_prev = im

    print(f"Frame {i} matches {matched_db.shape[0]}")

    E, mask = get_essential_matrix(matched_db, matched_query, K)
    p0_in, p1_in = matched_db[mask], matched_query[mask]

    R, T = get_pose(E, p0_in, p1_in, K)

    pose_history.append((R, R.dot(T)))

    cloud = triangulate(p0_in, p1_in, R, T, K)
    plot_frame_shift(ax_image, im, p0_in, p1_in)

    plot_camera_pose(ax_full, pose_history)
    plot_landmarks(ax_last20, cloud, pose_history)

    if lkt:
        # update keypoints from the last image
        p_prev = p1_in.reshape((-1, 1, 2)).astype(np.float32)

    # wait for rendering
    plt.pause(1e-6)

# TODO: use ground truth
