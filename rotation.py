import cv2

import numpy as np



def find_matches_using_orb(image1, image2):
    # Convert images to grayscale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(image1_gray, None)
    kp2, des2 = orb.detectAndCompute(image2_gray, None)

    # Create a brute-force matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors using brute-force matcher
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    return matches, kp1, kp2

# Load the original and transformed images
image1 = cv2.imread('C:/Users/klam0/Documents/University/code/datasets/val/images/ORION_orig_1_val.png')
image2 = cv2.imread('C:/Users/klam0/Documents/University/code/datasets/val/images/random.png')


# Find matches using ORB
matches, kp1, kp2 = find_matches_using_orb(image1, image2)

print("Number of matches:", len(matches))


# Draw the matches
result = cv2.drawMatches(
    image1, kp1, image2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Display the matches
cv2.imshow("Matches", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

