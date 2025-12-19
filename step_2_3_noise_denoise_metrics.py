import cv2
import numpy as np
import os
import math

os.makedirs("outputs/2_3", exist_ok=True)

# --- Metrikler ---
def mse(img1, img2):
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

def psnr(img1, img2):
    m = mse(img1, img2)
    if m == 0:
        return float("inf")
    PIXEL_MAX = 255.0
    return 10.0 * math.log10((PIXEL_MAX ** 2) / m)

# --- Gürültüler ---
def add_gaussian_noise(gray, mean=0, sigma=15):
    noise = np.random.normal(mean, sigma, gray.shape).astype(np.float32)
    noisy = gray.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper(gray, amount=0.02, salt_vs_pepper=0.5):
    noisy = gray.copy()
    h, w = gray.shape
    num_pixels = int(amount * h * w)

    # salt
    num_salt = int(num_pixels * salt_vs_pepper)
    ys = np.random.randint(0, h, num_salt)
    xs = np.random.randint(0, w, num_salt)
    noisy[ys, xs] = 255

    # pepper
    num_pepper = num_pixels - num_salt
    ys = np.random.randint(0, h, num_pepper)
    xs = np.random.randint(0, w, num_pepper)
    noisy[ys, xs] = 0

    return noisy

# --- Filtreler ---
def mean_filter(gray, k=3):
    return cv2.blur(gray, (k, k))

def gaussian_filter(gray, k=5, sigma=1.0):
    return cv2.GaussianBlur(gray, (k, k), sigma)

def median_filter(gray, k=5):
    return cv2.medianBlur(gray, k)

# --- İşlem akışı ---
image_paths = ["outputs/resized_img1.jpg", "outputs/resized_img2.jpg"]

results = []  # rapor tablosu için

for idx, path in enumerate(image_paths, start=1):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(f"outputs/2_3/img{idx}_gray.jpg", gray)

    # 1) Gürültü ekle
    g_noisy = add_gaussian_noise(gray, sigma=15)
    sp_noisy = add_salt_pepper(gray, amount=0.02)

    cv2.imwrite(f"outputs/2_3/img{idx}_gaussian_noisy.jpg", g_noisy)
    cv2.imwrite(f"outputs/2_3/img{idx}_saltpepper_noisy.jpg", sp_noisy)

    # 2) Gürültü giderme - Gaussian gürültüsü için
    g_mean = mean_filter(g_noisy, k=3)
    g_gauss = gaussian_filter(g_noisy, k=5, sigma=1.0)
    g_median = median_filter(g_noisy, k=5)

    cv2.imwrite(f"outputs/2_3/img{idx}_gaussNoise_mean.jpg", g_mean)
    cv2.imwrite(f"outputs/2_3/img{idx}_gaussNoise_gaussian.jpg", g_gauss)
    cv2.imwrite(f"outputs/2_3/img{idx}_gaussNoise_median.jpg", g_median)

    results.append(("img"+str(idx), "Gaussian", "Noisy", mse(gray, g_noisy), psnr(gray, g_noisy)))
    results.append(("img"+str(idx), "Gaussian", "Mean", mse(gray, g_mean), psnr(gray, g_mean)))
    results.append(("img"+str(idx), "Gaussian", "GaussianBlur", mse(gray, g_gauss), psnr(gray, g_gauss)))
    results.append(("img"+str(idx), "Gaussian", "Median", mse(gray, g_median), psnr(gray, g_median)))

    # 3) Gürültü giderme - Tuz-biber gürültüsü için
    sp_mean = mean_filter(sp_noisy, k=3)
    sp_gauss = gaussian_filter(sp_noisy, k=5, sigma=1.0)
    sp_median = median_filter(sp_noisy, k=5)

    cv2.imwrite(f"outputs/2_3/img{idx}_spNoise_mean.jpg", sp_mean)
    cv2.imwrite(f"outputs/2_3/img{idx}_spNoise_gaussian.jpg", sp_gauss)
    cv2.imwrite(f"outputs/2_3/img{idx}_spNoise_median.jpg", sp_median)

    results.append(("img"+str(idx), "SaltPepper", "Noisy", mse(gray, sp_noisy), psnr(gray, sp_noisy)))
    results.append(("img"+str(idx), "SaltPepper", "Mean", mse(gray, sp_mean), psnr(gray, sp_mean)))
    results.append(("img"+str(idx), "SaltPepper", "GaussianBlur", mse(gray, sp_gauss), psnr(gray, sp_gauss)))
    results.append(("img"+str(idx), "SaltPepper", "Median", mse(gray, sp_median), psnr(gray, sp_median)))

# 4) Sonuçları txt olarak kaydet (rapora tabloyu buradan dolduracaksın)
with open("outputs/2_3/metrics_table.txt", "w", encoding="utf-8") as f:
    f.write("Image\tNoise\tMethod\tMSE\tPSNR(dB)\n")
    for r in results:
        f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.2f}\t{r[4]:.2f}\n")

print("2.3 tamamlandı ")
print("Tablo: outputs/2_3/metrics_table.txt")
