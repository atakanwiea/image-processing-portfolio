import cv2
import numpy as np
import os

os.makedirs("outputs/2_8", exist_ok=True)

# 4 görüntüyü buradan alıyoruz (senin akışına uygun)
paths = [
    "outputs/resized_img1.jpg",
    "outputs/resized_img2.jpg",
    "outputs/resized_img3.jpg",
    "outputs/resized_img4.jpg",
]

imgs = []
for p in paths:
    im = cv2.imread(p)
    if im is None:
        raise FileNotFoundError(f"Okunamadi: {p}")
    imgs.append(im)

def stitch_pair(base, add, tag="12", nfeatures=1500, ratio=0.75, ransac_thresh=4.0):
    """
    base: mevcut panorama (BGR)
    add : eklenecek görüntü (BGR)
    tag : çıktı isimleri için etiket
    """
    grayA = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(add, cv2.COLOR_BGR2GRAY)

    # 1) ORB keypoint + descriptor
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kpA, desA = orb.detectAndCompute(grayA, None)
    kpB, desB = orb.detectAndCompute(grayB, None)

    if desA is None or desB is None or len(kpA) < 10 or len(kpB) < 10:
        raise RuntimeError("Yeterli anahtar nokta bulunamadi (ORB).")

    # 2) KNN eşleştirme + ratio test (descriptor tabanlı eşleştirme)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(desA, desB, k=2)

    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 10:
        raise RuntimeError(f"Yeterli iyi eşleşme yok: {len(good)}")

    # RANSAC öncesi eşleşme görseli (ilk 60)
    raw_vis = cv2.drawMatches(
        base, kpA, add, kpB, sorted(good, key=lambda x: x.distance)[:60],
        None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(f"outputs/2_8/matches_raw_{tag}.jpg", raw_vis)

    # 3) Homografi için noktaları hazırla
    ptsA = np.float32([kpA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ptsB = np.float32([kpB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 4) RANSAC ile homografi
    H, mask = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, ransac_thresh)
    if H is None:
        raise RuntimeError("Homografi bulunamadi (H None).")

    inliers = mask.ravel().tolist()

    # RANSAC sonrası (inlier) eşleşme görseli
    inlier_matches = [m for m, keep in zip(good, inliers) if keep]
    ransac_vis = cv2.drawMatches(
        base, kpA, add, kpB, sorted(inlier_matches, key=lambda x: x.distance)[:60],
        None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(f"outputs/2_8/matches_ransac_{tag}.jpg", ransac_vis)

    # 5) Warp için çıktı tuvali hesapla (base + add sığsın)
    hA, wA = base.shape[:2]
    hB, wB = add.shape[:2]

    cornersB = np.float32([[0,0],[wB,0],[wB,hB],[0,hB]]).reshape(-1,1,2)
    cornersB_warp = cv2.perspectiveTransform(cornersB, H)

    cornersA = np.float32([[0,0],[wA,0],[wA,hA],[0,hA]]).reshape(-1,1,2)

    all_corners = np.vstack((cornersA, cornersB_warp))
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    tx = -xmin
    ty = -ymin

    T = np.array([[1,0,tx],
                  [0,1,ty],
                  [0,0, 1]], dtype=np.float64)

    out_w = xmax - xmin
    out_h = ymax - ymin

    # 6) add görüntüsünü warp et
    warped_add = cv2.warpPerspective(add, T @ H, (out_w, out_h))
    cv2.imwrite(f"outputs/2_8/warped_image_{tag}.jpg", warped_add)

    # 7) base görüntüsünü tuvale yerleştir
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[ty:ty+hA, tx:tx+wA] = base

    # 8) Basit blending (mask ile)
    mask_canvas = (canvas.sum(axis=2) > 0).astype(np.uint8)
    mask_warped = (warped_add.sum(axis=2) > 0).astype(np.uint8)

    overlap = (mask_canvas & mask_warped).astype(np.uint8)
    only_canvas = (mask_canvas & (1 - overlap)).astype(np.uint8)
    only_warped = (mask_warped & (1 - overlap)).astype(np.uint8)

    result = np.zeros_like(canvas)
    result[only_canvas == 1] = canvas[only_canvas == 1]
    result[only_warped == 1] = warped_add[only_warped == 1]

    # overlap bölgesinde ortalama al (basit ama iş görür)
    if overlap.sum() > 0:
        ov_idx = overlap == 1
        blended = (0.5 * canvas[ov_idx].astype(np.float32) + 0.5 * warped_add[ov_idx].astype(np.float32))
        result[ov_idx] = blended.astype(np.uint8)

    return result

# --- 4 görüntüyü sırayla birleştir ---
pan = imgs[0]

pan = stitch_pair(pan, imgs[1], tag="12")
cv2.imwrite("outputs/2_8/pan_step12.jpg", pan)

pan = stitch_pair(pan, imgs[2], tag="123")
cv2.imwrite("outputs/2_8/pan_step123.jpg", pan)

pan = stitch_pair(pan, imgs[3], tag="1234")
cv2.imwrite("outputs/2_8/final_panorama.jpg", pan)

print("2.8 tamamlandi. Ciktilar: outputs/2_8/")
