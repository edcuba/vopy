import numpy as np
from os import path

import skimage
import skimage.io
import skimage.transform


def load_dataset(dataset_path, image_ratio=1):
    ground_truth = np.loadtxt(path.join(dataset_path, "poses.txt"))
    K = np.loadtxt(path.join(dataset_path, "K.txt"))

    def load_image(frame):
        im_path = path.join(dataset_path, "images", "img_%05d.png" % frame)
        full_image = skimage.io.imread(im_path, as_gray=True)
        return skimage.transform.rescale(full_image, image_ratio)

    return load_image, ground_truth, K
