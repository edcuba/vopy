## Visual Odometry pipeline in Python

Vision Algorithms for Mobile Robotics UZH 2020.
- Author: Eduard Čuba
- Course website: http://rpg.ifi.uzh.ch/teaching.html

### Demo
- Parking: https://youtu.be/FmRUN0YJZEI
- KITTI: https://youtu.be/oRbUsLNJslQ

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

To use the Kitti dataset, unpack it into `data/kitti` and make sure the following paths are valid:
- data/kitti/00/00xxxx.png (for images)
- data/kitti/K.txt (camera calibration matrix in a NumPy readable format - see parking/K.txt)

## Running

From the project directory
```
. venv/bin/activate   # make sure the Python env is activated
python -m vopy.main
```

For the run configuration, see variables `kitti` and `lkt` in `main.py` for the Kitti dataset and Lucas-Kanade Tracker respectively.

# About

Two VO pipelines are implemented within the solution
- simple pipeline using custom feature extraction (Harris), selection, and matching (see `keypoints.py`)
- Lucas-Kanade tracker pipeline using LKT implementation and features from OpenCV

## Steps

A brief overview of the implemented pipeline:

### Feature extraction

I used the Harris corner detector to find feasible features to track across the frames.
A Python implementation of reference Matlab code based on Sobel filters and convolution is employed.

Best N corners are selected using non-maxima suppression and described by their pixel neighborhoods.
For the Lucas-Kanade, features from `cv2.goodFeaturesToTrack` are used.

### Feature tracking

For Harris corners, I used a simple brute-force approach for feature matching based on the code from practical exercises.
Euclidian distances between the descriptors are computed, then each point is assigned its closest match, duplicates are removed and keypoints matches are extracted from the matched descriptors.
For the Lucas-Kanade, features from the previous frame are tracked in the new frame using the `cv2.calcOpticalFlowPyrLK` method.

### Pose estimation

First, the Essential Matrix is estimated from the keypoint correspondences using the `cv2.findEssentialMat` method, which uses RANSAC internally.

Rotation and translation are recovered from the Essential Matrix using `cv2.recoverPose` call with inlier correspondences from RANSAC.

### Point cloud triangulation

3D landmarks are triangulated from the point correspondences and their projection matrices (K.dot(np.eye(3, 4)), K.dot(RT)) using `cv2.triangulatePoints` method. Returned 3D homogenous coordinates are dehomogenized, and the points projected behind the camera frame are rejected.

### Adding new landmarks

After some experimentation, I only ended up with a simple system for adding new landmarks. When only using Harris and keypoint matching, keypoints are extracted from each frame and matched between adjacent ones.

In the LKT enabled version, keypoints are extracted from a single frame and then propagated through the next frames. Once the number of available keypoints to track is lower than a threshold (250), new keypoints are extracted.  The whole set of the previous keypoints is replaced by a newly extracted one. Keeping the old ones has often concentrated the accumulated keypoints in a single region (even after removing duplicates) - additional non-max suppression could help here.

## Encountered problems

#### Floating-point issues with triangulation:
When attempting to implement the 3D point triangulation by myself, I ran into problems with floating-point matrix operations. The triangulation test was reporting a large error, and I couldn't reproduce the Matlab reference implementation. After a lot of debugging, I got a different result from a dot product of the same input matrixes as in Matlab; after wasting a couple of hours here, I switched to OpenCV.

#### Data types in OpenCV
OpenCV has issues with natively handling NumPy arrays of floats with not strictly defined types. That led to segmentation faults. Casting NumPy arrays with keypoints using `.astype(np.float32)` solved the problem.

#### Lots of jitter when the car turns KITTI
Good results heavily depend on parameter tuning. I didn't find a fully satisfying configuration. Further non-linear adjustment and a lot of work on triangulating new landmarks are probably required to fix this.

#### Parameter tuning
Getting good results requires a lot of play with all the parameters under the hood. I found a usable configuration, yet it's far from perfect.

#### Python
I chose to implement the project in Python because I'm much more confident in it than in Matlab. It makes understanding the project structure and functions much easier for me. However, as all the reference scripts are in Matlab, I spent an unreasonable amount of time solving dull issues, rewriting some parts of code, looking for equivalent methods, and verifying my implementation when debugging errors.

## Conclusion

The implemented solution works well on a simple dataset like the parking garage one. Getting good results on a dataset like Kitti would require further improvements to triangulating new landmarks, tracking them across the frames, and possibly implementing non-linear adjustment.

My motivation for working on the project was to better understand how visual odometry works and have a try to implement it myself - and run into all the errors caused by only skimming through the project materials. To that end, I'm satisfied with the results, although the project would need much more time to make it work flawlessly.
