import cv2
import os

print("Kod basladi")

os.makedirs("outputs/2_8", exist_ok=True)

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
        print("OKUNAMADI:", p)
        exit()
    imgs.append(im)

print("Goruntuler yuklendi")

stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

status, pano = stitcher.stitch(imgs)

print("STATUS:", status)

if status == cv2.Stitcher_OK:
    cv2.imwrite("outputs/2_8/final_panorama.jpg", pano)
    print("OK -> outputs/2_8/final_panorama.jpg")
else:
    print("Stitcher basarisiz")
