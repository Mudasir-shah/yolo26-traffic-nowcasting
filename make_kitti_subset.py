"""Create the 800-train / 200-val KITTI subset used for CPU fine-tuning (symlinks, seed 0).

First download KITTI in YOLO format (Ultralytics release, 390 MB):
    curl -L -o kitti.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/kitti.zip && unzip kitti.zip -d datasets/kitti
"""
import os, random, sys, collections

SRC = sys.argv[1] if len(sys.argv) > 1 else "datasets/kitti"
DST = sys.argv[2] if len(sys.argv) > 2 else "datasets/kitti_sub"
random.seed(0)
for split, n in [("train", 800), ("val", 200)]:
    ims = sorted(os.listdir(f"{SRC}/images/{split}")); pick = random.sample(ims, n)
    for sub in ("images", "labels"): os.makedirs(f"{DST}/{sub}/{split}", exist_ok=True)
    for im in pick:
        stem = os.path.splitext(im)[0]
        os.symlink(os.path.abspath(f"{SRC}/images/{split}/{im}"), f"{DST}/images/{split}/{im}")
        os.symlink(os.path.abspath(f"{SRC}/labels/{split}/{stem}.txt"), f"{DST}/labels/{split}/{stem}.txt")
names = ["car", "van", "truck", "pedestrian", "person_sitting", "cyclist", "tram", "misc"]
open("kitti_sub.yaml", "w").write(f"path: {os.path.abspath(DST)}\ntrain: images/train\nval: images/val\nnames:\n" +
                                  "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
for split in ("train", "val"):
    c = collections.Counter()
    for f in os.listdir(f"{DST}/labels/{split}"):
        for line in open(f"{DST}/labels/{split}/{f}"): c[names[int(line.split()[0])]] += 1
    print(split, dict(c.most_common()))
