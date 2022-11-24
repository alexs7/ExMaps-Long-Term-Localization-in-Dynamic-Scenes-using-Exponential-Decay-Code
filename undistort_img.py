# from https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Extended-CMU-Seasons/
# c0 OPENCV 1024 768 868.993378 866.063001 525.942323 420.042529 -0.399431 0.188924 0.000153 0.000571
# c1 OPENCV 1024 768 873.382641 876.489513 529.324138 397.272397 -0.397066 0.181925 0.000176 -0.000579

import numpy as np
import cv2

def undistort_cmu(img):
    distortion_params = np.array([-0.399431, 0.188924, 0.000153, 0.000571])
    fx = 868.993378
    fy = 866.063001
    cx = 525.942323
    cy = 420.042529
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    undst = cv2.undistort(img, K, distortion_params, None, K)
    return undst