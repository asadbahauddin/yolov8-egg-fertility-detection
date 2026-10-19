import os, glob
from collections import Counter

def count_labels(labels_dir):
    cls_cnt = Counter()
    images = set()
    total_boxes = 0
    for p in glob.glob(os.path.join(labels_dir, "*.txt")):
        img_stem = os.path.splitext(os.path.basename(p))[0]
        images.add(img_stem)
        with open(p, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    cls_cnt[cls] += 1
                    total_boxes += 1
    return cls_cnt, len(images), total_boxes

splits = {
    "train": "train/labels",
    "val": "valid/labels",
    "test": "test/labels",
}

names = {0: "fertile", 1: "infertile"}  # pastikan sesuai data.yaml

for split, d in splits.items():
    cls_cnt, num_imgs, total_boxes = count_labels(d)
    print(f"\n[{split.upper()}]")
    print(f"  images: {num_imgs}")
    print(f"  total boxes: {total_boxes}  (avg {total_boxes/max(1,num_imgs):.2f} per image)")
    for k in sorted(cls_cnt):
        print(f"  {names.get(k, k)}: {cls_cnt[k]}  ({(cls_cnt[k]/max(1,total_boxes))*100:.2f}%)")
