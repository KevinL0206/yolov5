import cv2
import numpy as np
import math


# Load the two images
img_original = cv2.imread("URSA_orig_3_val.png")
img_rotated = cv2.imread("URSA_1_angle_30_elevation_0.75_3_val.png")

# Convert images to grayscale
gray_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
gray_rotated = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2GRAY)

# Create a SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for the two images
kp_original, des_original = sift.detectAndCompute(gray_original, None)
kp_rotated, des_rotated = sift.detectAndCompute(gray_rotated, None)

# Create a BFMatcher object
bf = cv2.BFMatcher()

# Match the keypoints of the two images
matches = bf.match(des_original, des_rotated)

# Sort the matches by distance
matches = sorted(matches, key=lambda x:x.distance)

# Find the transformation matrix
src_pts = np.float32([ kp_rotated[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
dst_pts = np.float32([ kp_original[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)

# Warp the rotated image using the transformation matrix
img_rotated_warped = cv2.warpPerspective(img_rotated, M, (img_original.shape[1], img_original.shape[0]))

# Combine the two images
result = cv2.addWeighted(img_original, 1, img_rotated_warped, 1, 0)

# Display the result
cv2.imshow('Mapped stars', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
