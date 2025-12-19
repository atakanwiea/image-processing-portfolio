import cv2
import numpy as np
import os

os.makedirs("outputs/2_4", exist_ok=True)

# Girdi
img = cv2.imread("outputs/resized_img1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("outputs/2_4/original_gray.jpg", gray)

# -------- 1) Laplacian tabanlı keskinleştirme --------
# Laplacian (kenar) -> orijinale ekleyerek keskinleştirme
lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
lap_abs = cv2.convertScaleAbs(lap)
lap_sharp = cv2.addWeighted(gray, 1.0, lap_abs, 0.7, 0)  # ağırlık = 0.7
cv2.imwrite("outputs/2_4/laplacian_edges.jpg", lap_abs)
cv2.imwrite("outputs/2_4/laplacian_sharp_w0_7.jpg", lap_sharp)

# Parametre değişimi (Laplacian ağırlığı)
for w in [0.3, 0.7, 1.2]:
    sharp = cv2.addWeighted(gray, 1.0, lap_abs, w, 0)
    cv2.imwrite(f"outputs/2_4/laplacian_sharp_w{str(w).replace('.','_')}.jpg", sharp)

# -------- 2) Unsharp Masking --------
# Unsharp = Orijinal + k*(Orijinal - Blur)
blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
mask = cv2.subtract(gray, blur)

for k in [0.5, 1.0, 1.5]:
    unsharp = cv2.addWeighted(gray, 1.0, mask, k, 0)
    cv2.imwrite(f"outputs/2_4/unsharp_k{str(k).replace('.','_')}.jpg", unsharp)

cv2.imwrite("outputs/2_4/unsharp_blur.jpg", blur)
cv2.imwrite("outputs/2_4/unsharp_mask.jpg", mask)

# -------- 3) High-Boost Filtering --------
# High-boost = A*Orijinal - Blur  (A>1)
# A arttıkça keskinlik artar ama gürültü/halo da artabilir
for A in [1.2, 1.8, 2.5]:
    highboost = cv2.addWeighted(gray, A, blur, -1.0, 0)
    cv2.imwrite(f"outputs/2_4/highboost_A{str(A).replace('.','_')}.jpg", highboost)

print("2.4 tamamlandi. Ciktilar: outputs/2_4/")
