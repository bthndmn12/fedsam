import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read input image
# img = cv2.imread('D:\\fedsam\\fedsamv1\\images\\w1.jpg')

img = cv2.imread("D:\\fedsam\\fedsamv1\\images\\w2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 3)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
blob = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)

blob = (255-blob)

contours = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
big_contour = max(contours, key=cv2.contourArea)

blob_area_thresh = 100
blob_area = cv2.contourArea(big_contour)
if blob_area > blob_area_thresh:
    print('blob area:', blob_area)

result = img.copy()
cv2.drawContours(result, [big_contour], -1, (0,0,255), 1)

cv2.imshow('thresh', thresh)
cv2.imshow('blob', blob)
cv2.imshow('result', result)
cv2.waitKey()

# # Check if image is loaded correctly
# if im is None:
#     print("Could not open or find the image")
# else:
#     # Set up the detector with default parameters.
# # Set up the detector with default parameters.
#     detector = cv2.SimpleBlobDetector_create() 
#     # Detect blobs.
#     keypoints = detector.detect(im)
 
#     # Draw detected blobs as red circles.
#     # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
#     im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
#     # Show keypoints
#     cv2.imshow("Keypoints", im_with_keypoints)
#     cv2.waitKey(0)
# # # Convert to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Blur the image to reduce noise
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# # Compute Horizontal Gradient
# gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)

# # Absolute Gradient
# absGradX = cv2.convertScaleAbs(gradX)

# # Threshold the Image
# threshImg = cv2.adaptiveThreshold(absGradX, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
#                    cv2.THRESH_BINARY, blockSize=9, C=4)

# # Display Result
# plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(threshImg,cmap='gray'),plt.title('Edge Detected')
# plt.xticks([]), plt.yticks([])
# plt.show()

