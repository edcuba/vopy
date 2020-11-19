import numpy as np
from os import path
import skimage as ski

DATASET_PATH = "data/parking"
LAST_FRAME = 598

def load_dataset(dataset_path):
    ground_truth = np.loadtxt(path.join(dataset_path, "poses.txt"))   
    K = np.loadtxt(path.join(dataset_path, "K.txt"))

    def load_image(frame):
    	im_path = path.join(dataset_path, "images", "img_%05d.png" % frame)
    	return ski.io.imread(im_path, as_gray=True)
    return load_image, ground_truth, K


images, ground_truth, K = load_dataset(DATASET_PATH)

print(ground_truth)
print(K)


"""
%% Setup
ds = 2; % 0: KITTI, 1: Malaga, 2: parking

% Path containing images, depths and all...
last_frame = 598;
K = load([parking_path '/K.txt']);
ground_truth = load([parking_path '/poses.txt']);
ground_truth = ground_truth(:, [end-8 end]);

img0 = rgb2gray(imread([parking_path ...
    sprintf('/images/img_%05d.png',bootstrap_frames(1))]));
img1 = rgb2gray(imread([parking_path ...
    sprintf('/images/img_%05d.png',bootstrap_frames(2))]));

%% Continuous operation
range = (bootstrap_frames(2)+1):last_frame;
for i = range
    fprintf('\n\nProcessing frame %d\n=====================\n', i);
    image = im2uint8(rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png',i)])));
    % Makes sure that plots refresh.    
    pause(0.01);
    prev_img = image;
end
"""
