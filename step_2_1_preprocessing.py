import cv2
import numpy as np
import os

image_dir = "images"
base_names = ["img1", "img2", "img3", "img4"]

def find_image_path(base_name):
    for ext in [".jpg", ".jpeg", ".png"]:
        path = os.path.join(image_dir, base_name + ext)
        if os.path.exists(path):
            return path
    return None

images = []

print("=== GÖRÜNTÜ TEMEL ÖZELLİKLERİ ===\n")

for base in base_names:
    path = find_image_path(base)

    if path is None:
        print(f"{base} bulunamadı!")
        continue

    img = cv2.imread(path)

    height, width, channels = img.shape
    dtype = img.dtype
    min_val = img.min()
    max_val = img.max()

    print(f"Görüntü: {os.path.basename(path)}")
    print(f"Boyut         : {width} x {height}")
    print(f"Kanal Sayısı  : {channels}")
    print(f"Veri Tipi     : {dtype}")
    print(f"Dinamik Aralık: {min_val} - {max_val}\n")

    images.append(img)

# Ortak boyuta getirme (ilk görüntü referans)
ref_height, ref_width = images[0].shape[:2]

os.makedirs("outputs", exist_ok=True)

for i, img in enumerate(images):
    resized = cv2.resize(img, (ref_width, ref_height))
    cv2.imwrite(f"outputs/resized_img{i+1}.jpg", resized)

print("Tüm görüntüler ortak boyuta getirildi.")
