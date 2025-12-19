import cv2
import numpy as np

img = cv2.imread("outputs/2_8/final_panorama.jpg")
if img is None:
    raise FileNotFoundError("outputs/2_8/final_panorama.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# siyah alanları maskele
mask = gray > 5
coords = np.column_stack(np.where(mask))
y0, x0 = coords.min(axis=0)
y1, x1 = coords.max(axis=0)

cropped = img[y0:y1+1, x0:x1+1]
cv2.imwrite("outputs/2_8/final_panorama_cropped.jpg", cropped)

print("OK -> outputs/2_8/final_panorama_cropped.jpg")
