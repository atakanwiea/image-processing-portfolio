import cv2
import numpy as np
import os

os.makedirs("outputs/2_5", exist_ok=True)

# Girdi
img = cv2.imread("outputs/resized_img1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("outputs/2_5/original_gray.jpg", gray)

# -------- Sobel --------
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_mag = cv2.convertScaleAbs(sobel_mag)

cv2.imwrite("outputs/2_5/sobel_edges.jpg", sobel_mag)

# -------- Prewitt --------
kernel_x = np.array([[1,0,-1],
                     [1,0,-1],
                     [1,0,-1]])

kernel_y = np.array([[1,1,1],
                     [0,0,0],
                     [-1,-1,-1]])

prewitt_x = cv2.filter2D(gray, -1, kernel_x)
prewitt_y = cv2.filter2D(gray, -1, kernel_y)
prewitt_mag = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

cv2.imwrite("outputs/2_5/prewitt_edges.jpg", prewitt_mag)

# -------- Canny --------
canny_low = cv2.Canny(gray, 50, 150)
canny_high = cv2.Canny(gray, 100, 200)

cv2.imwrite("outputs/2_5/canny_50_150.jpg", canny_low)
cv2.imwrite("outputs/2_5/canny_100_200.jpg", canny_high)

print("2.5 tamamlandi. Ciktilar: outputs/2_5/")
