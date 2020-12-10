import numpy as np
from scipy import signal, spatial

# Parameters

sobel_y = np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
sobel_x = sobel_y.T


def harris(img, patch_size=9, kappa=0.08):
    Ix = signal.convolve2d(sobel_x, img, 'valid')
    Iy = signal.convolve2d(sobel_y, img, 'valid')
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    patch = np.ones([patch_size, patch_size])
    pr = np.floor(patch_size / 2).astype(int)

    sIxx = signal.convolve2d(Ixx, patch, 'valid')
    sIyy = signal.convolve2d(Iyy, patch, 'valid')
    sIxy = signal.convolve2d(Ixy, patch, 'valid')

    scores = (sIxx * sIyy - sIxy ** 2) - kappa * (sIxx + sIyy) ** 2

    return np.pad(scores.clip(min=0), [1+pr, 1+pr])


def select_keypoints(scores, num_keypoints=200, nonmax_radius=8):
    r = nonmax_radius

    keypoints = np.zeros([num_keypoints, 2])
    temp_scores = np.pad(scores, [r, r])
    zero_patch = np.zeros([2*r + 1, 2*r + 1])

    for i in range(num_keypoints):
        kp = np.argmax(temp_scores, axis=None)
        idx = np.unravel_index(kp, temp_scores.shape)
        keypoints[i] = idx
        temp_scores[idx[0]-r:idx[0]+r+1, idx[1]-r:idx[1]+r+1] = zero_patch

    # correct for padding
    return (keypoints - r).astype(int)


def describe_keypoints(img, keypoints, desc_radius=9):
    # return image patches for the keypoints
    r = desc_radius
    padded = np.pad(img, [r, r])
    side = 2*r + 1
    new_size = side**2
    return [np.reshape(padded[x:x+side, y:y+side], new_size) for x, y in keypoints]


def match_descriptors(query_desc, database_desc, match_lambda=4):
    # dist(i,j) = dst(u=XA[i], v=XB[j])
    dists = spatial.distance.cdist(database_desc, query_desc, "euclidean")
    threshold = dists.min() * match_lambda
    mins = np.argmin(dists, axis=0)
    vals = dists[mins, range(dists.shape[1])]
    mins[vals > threshold] = -1
    v, idx = np.unique(mins, return_index=True)
    stacked = np.vstack([idx, v])
    return stacked[:,np.all(stacked != -1, axis=0)].T
