import cv2
import numpy as np
import os

os.makedirs("outputs/2_7", exist_ok=True)

# Girdi görüntüler (panorama için komşu 2 görüntü kullanıyoruz)
img1 = cv2.imread("outputs/resized_img1.jpg")
img2 = cv2.imread("outputs/resized_img2.jpg")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

cv2.imwrite("outputs/2_7/img1_gray.jpg", gray1)
cv2.imwrite("outputs/2_7/img2_gray.jpg", gray2)

# ========== 1) Köşe Tespiti (Harris) ==========
harris = cv2.cornerHarris(np.float32(gray1), 2, 3, 0.04)
harris = cv2.dilate(harris, None)

harris_vis = img1.copy()
harris_vis[harris > 0.01 * harris.max()] = [0, 0, 255]

cv2.imwrite("outputs/2_7/harris_corners.jpg", harris_vis)

# ========== 2) ORB – Anahtar Nokta & Tanımlayıcı ==========
def run_orb(nfeatures, tag):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0))
    img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0))

    cv2.imwrite(f"outputs/2_7/img1_orb_{tag}.jpg", img1_kp)
    cv2.imwrite(f"outputs/2_7/img2_orb_{tag}.jpg", img2_kp)

    # ========== 3) Descriptor Eşleştirme ==========
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    match_img = cv2.drawMatches(
        img1, kp1, img2, kp2, matches[:40], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite(f"outputs/2_7/orb_matches_{tag}.jpg", match_img)

    return len(kp1), len(matches)

# Parametre karşılaştırması
kp_500, m_500 = run_orb(500, "500")
kp_1000, m_1000 = run_orb(1000, "1000")

# Sonuçları txt olarak kaydet
with open("outputs/2_7/orb_stats.txt", "w", encoding="utf-8") as f:
    f.write("ORB Parametre Analizi\n")
    f.write(f"nfeatures=500  -> Keypoints: {kp_500}, Matches: {m_500}\n")
    f.write(f"nfeatures=1000 -> Keypoints: {kp_1000}, Matches: {m_1000}\n")

print("2.7 tamamlandi. Ciktilar: outputs/2_7/")
