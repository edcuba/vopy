# continuous operation of the pipeline

# a good resource for python: https://scikit-image.org/docs/dev/auto_examples/transform/plot_fundamental_matrix.html

# Import libraries
import cv2
import numpy as np

import matplotlib.pyplot as plt

from vopy.utils import load_dataset
from vopy.keypoints import harris, select_keypoints, describe_keypoints, match_descriptors, transform_coords
from vopy.pose import get_essential_matrix, get_pose, triangulate
from vopy.plotting import plot_frame_shift, plot_landmarks, plot_camera_pose, plot_matches

kitti = True
lkt = True

RADIUS = 9
LAMBDA = 4
KEYPOINTS = 300

if kitti:
    DATASET_PATH = "data/kitti/00"
    LAST_FRAME = 4540
    START_FRAME = 0
else:
    DATASET_PATH = "data/parking"
    LAST_FRAME = 598
    START_FRAME = 0

INIT_FRAME = START_FRAME + 2


if lkt:
    RADIUS = 7

lkt_flow = {
    "winSize": (2 * RADIUS + 1, 2 * RADIUS + 1),
    "maxLevel": 2,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
}

lkt_features = {
    "maxCorners": KEYPOINTS,
    "qualityLevel": 0.2,
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

T_history = []
R_history = []
match_history = []

fig = plt.figure(figsize=(14, 7))
ax_image = fig.add_subplot(2, 2, 1)

ax_matches = fig.add_subplot(2, 4, 5)

ax_full = fig.add_subplot(2, 4, 6)
ax_full.set_aspect('equal', 'datalim')

ax_last20 = fig.add_subplot(1, 2, 2)
ax_last20.set_aspect('equal', 'datalim')

# initialize the world frame
im_prev = get_image(START_FRAME)

if lkt:
    im_prev_t = im_prev.T # openCV is C++
    p_prev = cv2.goodFeaturesToTrack(im_prev_t, mask=None, **lkt_features)
else:
    k_prev = process_image(im_prev)
    d_prev = describe_keypoints(im_prev, k_prev, desc_radius=RADIUS)

R_total = np.eye(3, 3)
T_total = np.ones((3, 1))

for i in range(INIT_FRAME, LAST_FRAME + 1):
    im = get_image(i)

    if lkt:
        im_t = im.T # openCV is C++
        if p_prev.shape[0] < 200:
            # running out of features, add new keypoints to track
            p_new = cv2.goodFeaturesToTrack(im_prev_t, mask=None, **lkt_features)
            p_prev = np.unique(np.vstack((p_prev, p_new)), axis=-1)
        p_query, st, _ = cv2.calcOpticalFlowPyrLK(im_prev_t, im_t, p_prev, None, **lkt_flow)
        matched_db = p_prev[st == 1]
        matched_query = p_query[st == 1]
        im_prev_t = im_t

        # get array of dominant pixel motion for each pixel
        diff = abs(matched_db - matched_query).max(-1)
        mean_diff = diff.mean()

        if mean_diff < 3:
            # skip frame (car is stopped?)
            print(f"Frame {i} skipped, mean pixel difference too small: {mean_diff}")
            continue
    else:
        keypoints = process_image(im)
        descriptors = describe_keypoints(im, keypoints, desc_radius=RADIUS)

        # match descriptors between images
        dists = match_descriptors(descriptors, d_prev, match_lambda=LAMBDA)

        # matches in the query and db image
        matched_query, matched_db = keypoints[dists[:,0]], k_prev[dists[:,1]]
        k_prev, d_prev = keypoints, descriptors
        im_prev = im

    match_history.append((i, matched_query.shape[0]))
    plot_matches(ax_matches, match_history)

    E, mask = get_essential_matrix(matched_db, matched_query, K)
    p0_in, p1_in = matched_db[mask], matched_query[mask]

    R, T = get_pose(E, p0_in, p1_in, K)

    cloud = triangulate(p0_in, p1_in, R, T, K)
    plot_frame_shift(ax_image, im, p0_in, p1_in)

    T_total = T_total + R_total.dot(T)
    R_total = R.dot(R_total)

    R_history.append(R_total)
    T_history.append(T_total)

    plot_camera_pose(ax_full, T_history)
    plot_landmarks(ax_last20, cloud, T_history, R_total)

    if lkt:
        # update keypoints from the last image
        p_prev = p1_in.reshape((-1, 1, 2)).astype(np.float32)

    # wait for rendering
    plt.pause(1e-6)
