# continuous operation of the pipeline

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

import skimage.io as skio

from vopy.utils import load_dataset
from vopy.keypoints import harris, select_keypoints, describe_keypoints, match_descriptors

def process_image(im):
    harris_scores = harris(im)
    keypoints = select_keypoints(harris_scores)
    descriptors = describe_keypoints(im, keypoints)

    return harris_scores, keypoints, descriptors

DATASET_PATH = "../data/parking"
LAST_FRAME = 598

get_image, ground_truth, K = load_dataset(DATASET_PATH, image_ratio=1)

# initialize the world frame
im_0 = get_image(0)
h0, k0, d0 = process_image(im_0)

for i in range(1, LAST_FRAME+1):
    im = get_image(im)
    harris_scores, keypoints, descriptors = process_image(im)
