import base64
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

NEW_DATA_DIRS = [
    Path("/home/fumu/datadisk/标注区域划分/堆龙东嘎"),
    Path("/home/fumu/datadisk/标注区域划分/曲水"),
]
OLD_DATA_DIR = Path("/home/fumu/datadisk/drone_data_src")
DST_DIR = Path("/home/fumu/datadisk/building_seg_v5_mixed")

MIN_POINTS = 4

NEW_CLASS_MAP = {
    "residential": "building",
    "steel_roof": "building",
    "large_complex": "building",
    "glass_roof": "building",
    "under_construction": "building",
}
OLD_CLASS_MAP = {"building": "building", "car": "car"}
YOLO_CLASS_ID = {"building": 0, "car": 1}

MONTH_ORDER = ["Feb", "Mar", "Apr", "May", "Jun", "Jul", "July", "Aug", "Sep", "Oct", "Nov", "Dec"]
LOC_LIST = ["naiqiong", "yangda", "dongga", "gurong", "niedang", "qushui", "deqing"]


def convert_labelme_to_yolo(json_path, class_map):
    d = json.load(open(json_path))
    img_w = d["imageWidth"]
    img_h = d["imageHeight"]
    lines = []
    filtered = 0
    total = 0
    for shape in d["shapes"]:
        total += 1
        label = shape["label"]
        mapped = class_map.get(label)
        if mapped is None:
            filtered += 1
            continue
        points = shape["points"]
        if len(points) < MIN_POINTS:
            filtered += 1
            continue
        cls_id = YOLO_CLASS_ID[mapped]
        coords = []
        for p in points:
            x = max(0.0, min(1.0, p[0] / img_w))
            y = max(0.0, min(1.0, p[1] / img_h))
            coords.append(f"{x:.6f} {y:.6f}")
        lines.append(f"{cls_id} " + " ".join(coords))
    return lines, d, total, filtered


def find_or_extract_image(json_path, d, dst_img_dir):
    base = json_path.stem
    src_dir = json_path.parent
    for ext in [".PNG", ".png", ".jpg", ".jpeg", ".bmp", ".tif"]:
        candidate = src_dir / (base + ext)
        if candidate.exists():
            return candidate
    if d.get("imageData") is not None:
        img_data = base64.b64decode(d["imageData"])
        img_path = d.get("imagePath", base + ".PNG")
        ext = Path(img_path).suffix if img_path else ".PNG"
        if not ext:
            ext = ".PNG"
        out_path = dst_img_dir / (base + ext)
        with open(out_path, "wb") as f:
            f.write(img_data)
        return out_path
    return None


def get_location_group(filename, source):
    name = filename.replace(".json", "")
    if source == "old":
        LOC = "|".join(sorted(LOC_LIST, key=len, reverse=True))
        MON = "|".join(sorted(MONTH_ORDER, key=len, reverse=True))
        m = re.match(rf"(?:({MON}))?({LOC})(?:({MON}))?(.*)", name)
        if m:
            loc = m.group(2)
            num = m.group(4)
            return f"old_{loc}_{num}"
        return "old_other"
    else:
        return f"new_{Path(source).name}"


def main():
    random.seed(42)

    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    for sub in ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"]:
        (DST_DIR / sub).mkdir(parents=True, exist_ok=True)

    all_samples = []
    location_groups = defaultdict(list)
    label_counter = Counter()
    skipped_empty = 0
    total_filtered = 0
    total_shapes = 0

    for src_dir in NEW_DATA_DIRS:
        json_files = sorted([f for f in os.listdir(src_dir) if f.endswith(".json")])
        for jf in json_files:
            json_path = src_dir / jf
            lines, d, total, filtered = convert_labelme_to_yolo(json_path, NEW_CLASS_MAP)
            total_shapes += total
            total_filtered += filtered
            if not lines:
                skipped_empty += 1
                continue
            for line in lines:
                cls_id = int(line.split()[0])
                label_counter[cls_id] += 1
            grp = get_location_group(jf, str(src_dir))
            idx = len(all_samples)
            all_samples.append((json_path, lines, d))
            location_groups[grp].append(idx)

    json_files_old = sorted([f for f in os.listdir(OLD_DATA_DIR) if f.endswith(".json")])
    for jf in json_files_old:
        json_path = OLD_DATA_DIR / jf
        lines, d, total, filtered = convert_labelme_to_yolo(json_path, OLD_CLASS_MAP)
        total_shapes += total
        total_filtered += filtered
        if not lines:
            skipped_empty += 1
            continue
        for line in lines:
            cls_id = int(line.split()[0])
            label_counter[cls_id] += 1
        grp = get_location_group(jf, "old")
        idx = len(all_samples)
        all_samples.append((json_path, lines, d))
        location_groups[grp].append(idx)

    CLASS_NAMES = {0: "building", 1: "car"}
    print(f"总标注: {total_shapes}")
    print(f"过滤(<{MIN_POINTS}点): {total_filtered} ({total_filtered / total_shapes * 100:.1f}%)")
    print(f"保留: {total_shapes - total_filtered}")
    print(f"空标签: {skipped_empty}")
    print("\n类别:")
    for cls_id, count in label_counter.most_common():
        print(f"  {CLASS_NAMES[cls_id]}: {count}")

    group_list = sorted(location_groups.items())
    random.shuffle(group_list)

    n_groups = len(group_list)
    n_train = int(n_groups * 0.8)
    n_val = int(n_groups * 0.1)

    train_indices = []
    val_indices = []
    test_indices = []

    for i, (grp, indices) in enumerate(group_list):
        if i < n_train:
            train_indices.extend(indices)
        elif i < n_train + n_val:
            val_indices.extend(indices)
        else:
            test_indices.extend(indices)

    split_data = {
        "train": [all_samples[i] for i in train_indices],
        "val": [all_samples[i] for i in val_indices],
        "test": [all_samples[i] for i in test_indices],
    }

    print(f"\nTrain: {len(split_data['train'])}, Val: {len(split_data['val'])}, Test: {len(split_data['test'])}")

    skipped_noimg = 0
    for phase, samples in split_data.items():
        img_dir = DST_DIR / "images" / phase
        lbl_dir = DST_DIR / "labels" / phase
        for json_path, lines, d in samples:
            base = json_path.stem
            img_src = find_or_extract_image(json_path, d, img_dir)
            if img_src is None:
                skipped_noimg += 1
                continue
            img_dst = img_dir / img_src.name
            if not img_dst.exists() and img_src.parent != img_dir:
                shutil.copy(img_src, img_dst)
            label_dst = lbl_dir / (base + ".txt")
            with open(label_dst, "w") as f:
                f.write("\n".join(lines))

    print(f"\nDataset: {DST_DIR}")
    print(f"No image: {skipped_noimg}")
    for phase in ["train", "val", "test"]:
        n_img = len(list((DST_DIR / "images" / phase).iterdir()))
        n_lbl = len(list((DST_DIR / "labels" / phase).iterdir()))
        print(f"  {phase}: {n_img} images, {n_lbl} labels")


if __name__ == "__main__":
    main()
