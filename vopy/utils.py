import cv2
import numpy as np
from os import path


def load_dataset(dataset_path, image_ratio=1, kitti=False):
    K = np.loadtxt(path.join(dataset_path, "K.txt"))
    clahe = cv2.createCLAHE(clipLimit=5.0)

    def load_image(frame):
        subfolder = "image_0" if kitti else "images"
        img_name = "%06d.png" % frame if kitti else "img_%05d.png" % frame
        im_path = path.join(dataset_path, subfolder, img_name)
        full_image = cv2.imread(im_path, 0)
        return clahe.apply(full_image)

    return load_image, K
