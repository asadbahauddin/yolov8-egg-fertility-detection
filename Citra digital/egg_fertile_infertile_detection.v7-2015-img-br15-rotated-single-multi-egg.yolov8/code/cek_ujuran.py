import os, glob
import cv2
from collections import Counter

# Path ke folder dataset kamu
splits = ["test/images", "train/images", "valid/images"]

size_counter = Counter()

for split in splits:
    img_files = glob.glob(os.path.join(split, "*.jpg")) + glob.glob(os.path.join(split, "*.png"))
    for img_path in img_files:
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            size_counter[(w, h)] += 1

print("=== Daftar dimensi gambar & jumlahnya ===")
for (w, h), count in size_counter.most_common():
    print(f"{w}x{h} : {count} gambar")

print(f"\nTotal variasi ukuran: {len(size_counter)}")
