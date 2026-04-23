import json
import os
import re
import random
import shutil
from pathlib import Path
SRC_DIR = Path("/home/fumu/PycharmProjects/无人机分割数据集")
DST_DIR = Path("/home/fumu/PycharmProjects/building_dataset")
TRAIN_RATIO = 0.8
CLASS_MAP = {
    "building": 0,
    "car": 1,
}
def get_time_group(filename):
    name = filename.replace(".json", "")
    m = re.match(r"(dongga|naiqiong|yangda|gurong|niedang)(.*)", name)
    if m:
        location = m.group(1)
        suffix = m.group(2)
        m2 = re.match(r"(Feb|Mar|Apr|May|Jun|July|Jul|Aug|Sep|Oct|Nov|Dec)", suffix)
        if m2:
            return f"{location}_{m2.group(1)}"
        return f"{location}_unknown"
    m3 = re.match(r"(Feb|Mar|Apr|May|Jun|July|Jul|Aug|Sep|Oct|Nov|Dec)", name)
    if m3:
        return f"other_{m3.group(1)}"
    return "other"
def polygon_to_bbox(points, img_w, img_h):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    w = max(0, min(1, w))
    h = max(0, min(1, h))
    return cx, cy, w, h
def convert_one(json_path):
    data = json.load(open(json_path, "r"))
    img_w = data["imageWidth"]
    img_h = data["imageHeight"]
    lines = []
    for shape in data["shapes"]:
        label = shape["label"]
        if label not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[label]
        cx, cy, w, h = polygon_to_bbox(shape["points"], img_w, img_h)
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines
def find_image(json_path):
    jf = json_path.name
    base = jf.replace(".json", "")
    for ext in [".jpg", ".png", ".jpeg", ".bmp", ".tif"]:
        candidate = SRC_DIR / (base + ext)
        if candidate.exists():
            return candidate
    return None
def main():
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (DST_DIR / sub).mkdir(parents=True, exist_ok=True)
    json_files = sorted([f for f in os.listdir(SRC_DIR) if f.endswith(".json")])
    groups = {}
    for f in json_files:
        grp = get_time_group(f)
        groups.setdefault(grp, []).append(f)
    random.seed(42)
    train_files = []
    val_files = []
    for grp, files in sorted(groups.items()):
        random.shuffle(files)
        split_idx = int(len(files) * TRAIN_RATIO)
        train_files.extend(files[:split_idx])
        val_files.extend(files[split_idx:])
        print(f"  {grp}: total={len(files)}, train={split_idx}, val={len(files)-split_idx}")
    skipped = 0
    for phase, files in [("train", train_files), ("val", val_files)]:
        for jf in files:
            json_path = SRC_DIR / jf
            lines = convert_one(json_path)
            if not lines:
                skipped += 1
                continue
            img_src = find_image(json_path)
            if img_src is None:
                print(f"  找不到图片: {jf}")
                skipped += 1
                continue
            label_dst = DST_DIR / "labels" / phase / jf.replace(".json", ".txt")
            with open(label_dst, "w") as f:
                f.write("\n".join(lines))
            img_dst = DST_DIR / "images" / phase / img_src.name
            shutil.copy2(img_src, img_dst)
    print(f"\nTrain: {len(train_files)}, Val: {len(val_files)}, Skipped: {skipped}")
    print(f"Dataset created at: {DST_DIR}")
if __name__ == "__main__":
    main()