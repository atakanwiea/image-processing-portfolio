import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Girdi görüntüsü
img = cv2.imread("outputs/resized_img1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

os.makedirs("outputs/2_2", exist_ok=True)

# 1. Gri seviye histogram
plt.figure()
plt.hist(gray.ravel(), 256, [0,256])
plt.title("Gri Seviye Histogram")
plt.xlabel("Piksel Değeri")
plt.ylabel("Frekans")
plt.savefig("outputs/2_2/gray_histogram.png")
plt.close()

# 2. Parlaklık ve kontrast ayarı
alpha = 1.3  # kontrast
beta = 30    # parlaklık
brightness_contrast = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
cv2.imwrite("outputs/2_2/brightness_contrast.jpg", brightness_contrast)

# 3. Gamma düzeltme
def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

gamma_05 = gamma_correction(gray, 0.5)
gamma_15 = gamma_correction(gray, 1.5)

cv2.imwrite("outputs/2_2/gamma_0_5.jpg", gamma_05)
cv2.imwrite("outputs/2_2/gamma_1_5.jpg", gamma_15)

# 4. Histogram eşitleme
equalized = cv2.equalizeHist(gray)
cv2.imwrite("outputs/2_2/hist_equalized.jpg", equalized)

print("2.2 Histogram tabanlı işlemler tamamlandı.")
