## Visual Odometry pipeline in Python

Vision Algorithms for Mobile Robotics UZH 2020.
- Author: Eduard Čuba
- Course website: http://rpg.ifi.uzh.ch/teaching.html

### Installation

Python 3.7+ is required.

```
pip3 install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Preparing the data

Use the script `get-data.sh` to download and unpack the parking dataset.
```
./get-data.sh
```

To use Kitti dataset, unpack it into `data/kitti` and make sure the following paths are valid:
- data/kitti/00/00xxxx.png (for images)
- data/kitti/K.txt (camera calibration matrix in a numpy readable format - see parking/K.txt)

## Running

From the project directory
```
. venv/bin/activate   # make sure the Python env is activated
python -m vopy.main
```

For run configuration see variables `kitti` and `lkt` in `main.py` to use kitti dataset and Lucas-Kanade Tracker respectively.

# About

This is a simple Visual Odometry implementation in Python using some methods from OpenCV.

Two pipelines are implemented within the solution
- simple pipeline using custom feature extraction (Harris), selection and matching (see `keypoints.py`)
- Lucas-Kanade tracker pipeline using LKT implementation and features from OpenCV

## Steps

A brief overview of the implemented pipeline:

### Feature extraction

I used Harris corner detector to find good features for track across the frames.
A Python implementation of reference Matlab code using sobel filters and convolution is used.

Best N corners are selected using non-maxima supression and described by their pixel neighbourhoods.
For the Lucas-Kanade pipeline, features from `cv2.goodFeaturesToTrack` are used.

### Feature matching

For Harris corners, I used a simple brute-force approach for feature matching based on the code from practical exercises.
First, all euclidian distances between the descriptors are computed, then each point is assigned it's closest match and duplicates are removed.

For the Lucas-Kanade pipeline, features from the previous frame are tracked in the new frame using `cv2.calcOpticalFlowPyrLK` method.

## Encountered problems

Floating point issues with triangulation:
- when trying to write the 3D point triangulation by myself, I ran into problems with float numbers in matrices
    - triangulation test was reporting a huge error and I couldn't reproduce the Matlab reference implementation
    - after a lot of debugging, I was getting a different result from a dot product of the same input matrixes as in Matlab
    - after wasting a couple of hours here, I switched to OpenCV

Data types in OpenCV
- OpenCV has issues with natively handling numpy arrays of floats with not strictly defined types
- This led to segmentation faults, casting numpy arrays with `.astype(np.float32)` solved the problem

Parameter tuning
- getting good results requires a lot of play with all the parameters under the hood, I found a reasonable configuration but it's far from perfect

Lots of jitter in KITTI turns
- depends on the parameter tuning, but I didn't find a satisfactory configuration
- further non-linear adjustment and more work on triangulating new landmarks is probably required to fix this
