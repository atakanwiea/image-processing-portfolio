import cv2
import numpy as np
import os

os.makedirs("outputs/2_6", exist_ok=True)

# Girdi (renkli kullanacağız!)
img = cv2.imread("outputs/resized_img1.jpg")  # BGR okur
cv2.imwrite("outputs/2_6/original_bgr.jpg", img)

# ============== 1) HSV dönüşümü ==============
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv)

cv2.imwrite("outputs/2_6/hsv_H.jpg", H)
cv2.imwrite("outputs/2_6/hsv_S.jpg", S)
cv2.imwrite("outputs/2_6/hsv_V.jpg", V)

# V kanalında parlaklık düzeltme (gamma benzeri: ölçekleme)
# alpha > 1 => parlaklık artar, alpha < 1 => azalır
for alpha in [0.8, 1.2]:
    V_adj = np.clip(V.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    hsv_adj = cv2.merge([H, S, V_adj])
    bgr_from_hsv = cv2.cvtColor(hsv_adj, cv2.COLOR_HSV2BGR)
    cv2.imwrite(f"outputs/2_6/hsv_V_scaled_{str(alpha).replace('.','_')}.jpg", bgr_from_hsv)

# ============== 2) YCbCr dönüşümü ==============
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = cv2.split(ycrcb)

cv2.imwrite("outputs/2_6/ycbcr_Y.jpg", Y)
cv2.imwrite("outputs/2_6/ycbcr_Cr.jpg", Cr)
cv2.imwrite("outputs/2_6/ycbcr_Cb.jpg", Cb)

# Y kanalında parlaklık düzeltme
for alpha in [0.8, 1.2]:
    Y_adj = np.clip(Y.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    ycrcb_adj = cv2.merge([Y_adj, Cr, Cb])
    bgr_from_ycbcr = cv2.cvtColor(ycrcb_adj, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(f"outputs/2_6/ycbcr_Y_scaled_{str(alpha).replace('.','_')}.jpg", bgr_from_ycbcr)

print("2.6 tamamlandi. Ciktilar: outputs/2_6/")
