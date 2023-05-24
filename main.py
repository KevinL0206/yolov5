import torch
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os

os.chdir("c:/Users/klam0/Documents/University/code")
# Load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp49/weights/best.pt',force_reload=True)

# Set the model to evaluation mode
model.eval()

# Define the path to the image you want to make predictions on
image_path = 'C:/Users/klam0/Documents/University/nofilter.png'

# Define the kernel for morphological opening
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

# Load the image using OpenCV
img = cv2.imread(image_path)
#img = cv2.flip(img, 1)

# Create a gray layer with the same dimensions as the image
white_layer = np.ones_like(img) * 128

# Set the opacity of the white layer
opacity = 0.75

# Blend the white layer with the original image using the cv2.addWeighted() function
img = cv2.addWeighted(white_layer, opacity, img, 1 - opacity, 0)

# Display the resulting image with artificial fog
cv2.imshow('Artificial Fog', img)

# Convert the foggy image to YCrCb color space
ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# Split the channels: Y (luminance), Cr and Cb
y_channel, cr_channel, cb_channel = cv2.split(ycrcb_img)

# Create a CLAHE object (Arguments are optional)
clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))

# Apply CLAHE to the Y channel
clahe_y = clahe.apply(y_channel)

# Merge the CLAHE Y channel with the original Cr and Cb channels
enhanced_ycrcb = cv2.merge((clahe_y, cr_channel, cb_channel))

# Convert the enhanced YCrCb image back to the BGR color space
enhanced_img = cv2.cvtColor(enhanced_ycrcb, cv2.COLOR_YCrCb2BGR)

# Display the enhanced image
cv2.imshow('Enhanced Image', enhanced_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Create a red filter by setting red channel to 255 and green and blue channels to 0
red_filter = np.zeros_like(img)
red_filter[:,:,2] = 255
red_filter[:,:,1] = 0
red_filter[:,:,0] = 0

# Blend the red channel with the original image using the cv2.addWeighted() function
alpha = 0.7 # blending factor
img = cv2.addWeighted(img, alpha, red_filter, 1-alpha, 0)

# Display the resulting image with the red filter applied
cv2.imshow('Red Filter', img)


# Convert the image to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Set pixels with value above 250 to 255 in the V channel
hsv[:,:,2][hsv[:,:,2] > 250] = 255

# Decrease pixels with value below 250 by 80 in the V channel
hsv[:,:,2][hsv[:,:,2] <= 250] = np.maximum(hsv[:,:,2][hsv[:,:,2] <= 250] - 80, 0)

# Convert the image back to BGR
img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


#--Add Noise--
# Create a mask with a 10% chance of having a value of 128 for each pixel
mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
mask[np.random.random((img.shape[0], img.shape[1])) <= 0.1] = 128

# Replace the pixels in the original frame that are 128 in the mask with 128
img[mask == 128] = 128

# Merge the mask with the modified frame
merged = cv2.addWeighted(img, 1, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.5, 0)

cv2.imshow('noise', merged)
# Convert the blurred frame to grayscale
gray = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)

cv2.imshow('gray', gray)

# Perform contrast stretching
p1, p99 = np.percentile(gray, (1, 99))
stretched_gray = np.uint8(np.clip((gray - p1) / (p99 - p1) * 255.0, 0, 255))

cv2.imshow('stretched', stretched_gray)

# Apply Gaussian blur to the frame
blurred_frame = cv2.GaussianBlur(stretched_gray, (7,7), 0)

cv2.imshow('blur', blurred_frame)

# Apply a threshold to extract the bright pixels
thresh = cv2.threshold(blurred_frame, 245, 255, cv2.THRESH_BINARY)[1]

# Perform morphological opening to remove small bright regions
opened_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

cv2.imshow('open', opened_thresh)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find contours in the thresholded image
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_area = 0 # Set a minimum area threshold
max_area = 1000 # Set a maximum area threshold

# Create an array to store the 2D positions of the LEDs
led_positions = []

filtered_contours = []

# Draw bounding boxes around the detected contours and extract the centroid coordinates
for contour in contours:
    
    #Limit size of contours found
    area = cv2.contourArea(contour)
    if min_area <= area <= max_area:
        filtered_contours.append(contour)    

        x, y, w, h = cv2.boundingRect(contour)
        
        

        # Calculate the centroid of the contour
        centroid_x = (x + w/2)
        centroid_y = (y + h/2) 
        # Add the centroid coordinates to the array of LED positions
        led_positions.append((centroid_x, centroid_y))

print(led_positions)

"""
# Convert the image from OpenCV's BGR format to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert the image to a PIL Image object
img = Image.fromarray(img)
"""
# Make predictions on the image using the YOLOv5 model
results = model(opened_thresh)

# Print the predicted labels and bounding boxes
print(results.pandas().xyxy[0])

# Show the image with predicted labels and bounding boxes
rendered_image = np.squeeze(np.array(results.render()))
cv2.imshow('YOLOv5 predictions', rendered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
