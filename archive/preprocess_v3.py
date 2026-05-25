import json
import os
import re
import random
import base64
import shutil
from pathlib import Path
from collections import Counter, defaultdict

NEW_DATA_DIRS = [
    Path("/home/fumu/datadisk/标注区域划分/堆龙东嘎"),
    Path("/home/fumu/datadisk/标注区域划分/曲水"),
]
OLD_DATA_DIR = Path("/home/fumu/PycharmProjects/无人机分割数据集")
DST_DIR = Path("/home/fumu/datadisk/building_seg_dataset_v3")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
MIN_POINTS = 6

NEW_CLASS_MAP = {
    "residential": "building",
    "steel_roof": "building",
    "large_complex": "building",
    "glass_roof": "building",
    "under_construction": "building",
}
OLD_CLASS_MAP = {"building": "building"}
YOLO_CLASS_ID = {"building": 0}

MONTH_ORDER = ["Feb", "Mar", "Apr", "May", "Jun", "Jul", "July", "Aug", "Sep", "Oct", "Nov", "Dec"]
LOC_LIST = ["naiqiong", "yangda", "dongga", "gurong", "niedang", "qushui", "deqing"]
_LOC = "|".join(sorted(LOC_LIST, key=len, reverse=True))
_MONTH = "|".join(sorted(MONTH_ORDER, key=len, reverse=True))


def convert_labelme_to_yolo(json_path, class_map):
    d = json.load(open(json_path, "r"))
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
        coords = []
        for p in points:
            x = max(0.0, min(1.0, p[0] / img_w))
            y = max(0.0, min(1.0, p[1] / img_h))
            coords.append(f"{x:.6f} {y:.6f}")
        lines.append(f"{YOLO_CLASS_ID[mapped]} " + " ".join(coords))
    return lines, d, total, filtered


def find_or_extract_image(json_path, d, dst_img_dir):
    base = json_path.stem
    src_dir = json_path.parent
    for ext in [".PNG", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
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


def get_old_location_group(filename):
    name = filename.replace(".json", "").replace(".jpg", "").replace(".png", "")
    m = re.match(rf"(?:({_MONTH}))?({_LOC})(?:({_MONTH}))?(.*)", name)
    if m:
        loc = m.group(2)
        num = m.group(4)
        return f"old_{loc}_{num}"
    return "old_other"


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

    # New data: group by entire region (堆龙东嘎 or 曲水)
    for src_dir in NEW_DATA_DIRS:
        json_files = sorted([f for f in os.listdir(src_dir) if f.endswith(".json")])
        grp = f"new_{src_dir.name}"
        for jf in json_files:
            json_path = src_dir / jf
            lines, d, total, filtered = convert_labelme_to_yolo(json_path, NEW_CLASS_MAP)
            total_shapes += total
            total_filtered += filtered
            if not lines:
                skipped_empty += 1
                continue
            for line in lines:
                label_counter["building (new)"] += 1
            idx = len(all_samples)
            all_samples.append(("new", json_path, lines, d, str(src_dir)))
            location_groups[grp].append(idx)

    # Old data: group by location+number (e.g., naiqiong_10 includes Feb/Mar/July variants)
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
            label_counter["building (old)"] += 1
        grp = get_old_location_group(jf)
        idx = len(all_samples)
        all_samples.append(("old", json_path, lines, d, "old"))
        location_groups[grp].append(idx)

    print(f"总标注数: {total_shapes}")
    print(f"过滤(<{MIN_POINTS}点或类别不匹配): {total_filtered} ({total_filtered/total_shapes*100:.1f}%)")
    print(f"保留标注: {total_shapes - total_filtered}")
    print(f"过滤后空标签样本: {skipped_empty}")

    # Split by location groups - each group entirely in one split
    train_indices = []
    val_indices = []
    test_indices = []

    group_list = sorted(location_groups.items())
    random.shuffle(group_list)

    n_groups = len(group_list)
    n_train_groups = int(n_groups * TRAIN_RATIO)
    n_val_groups = int(n_groups * VAL_RATIO)
    n_test_groups = n_groups - n_train_groups - n_val_groups

    for i, (grp, indices) in enumerate(group_list):
        if i < n_train_groups:
            train_indices.extend(indices)
        elif i < n_train_groups + n_val_groups:
            val_indices.extend(indices)
        else:
            test_indices.extend(indices)

        is_multi = len(indices) > 1
        tag = "*" if is_multi else ""
        split_name = "train" if i < n_train_groups else ("val" if i < n_train_groups + n_val_groups else "test")
        print(f"  {grp}: total={len(indices)} -> {split_name} {tag}")

    # Verify no leakage
    splits_set = {
        "train": set(train_indices),
        "val": set(val_indices),
        "test": set(test_indices),
    }
    leakage_found = False
    for grp, indices in location_groups.items():
        present_splits = set()
        for idx in indices:
            for split_name, split_set in splits_set.items():
                if idx in split_set:
                    present_splits.add(split_name)
        if len(present_splits) > 1:
            print(f"  LEAK: {grp} -> {present_splits}")
            leakage_found = True

    if not leakage_found:
        print("\n数据泄漏检查: ✓ 无泄漏")

    split_data = {
        "train": [all_samples[i] for i in train_indices],
        "val": [all_samples[i] for i in val_indices],
        "test": [all_samples[i] for i in test_indices],
    }

    print(f"\nTotal valid samples: {len(all_samples)}")
    print(f"Train: {len(split_data['train'])}, Val: {len(split_data['val'])}, Test: {len(split_data['test'])}")
    print(f"\nLabel distribution:")
    for label, count in label_counter.most_common():
        print(f"  {label}: {count}")

    skipped_noimg = 0
    for phase, samples in split_data.items():
        img_dir = DST_DIR / "images" / phase
        lbl_dir = DST_DIR / "labels" / phase
        for src_type, json_path, lines, d, src_id in samples:
            base = json_path.stem
            img_src = find_or_extract_image(json_path, d, img_dir)
            if img_src is None:
                skipped_noimg += 1
                continue
            img_dst = img_dir / img_src.name
            if not img_dst.exists() and img_src.parent != img_dir:
                shutil.copy2(img_src, img_dst)
            label_dst = lbl_dir / (base + ".txt")
            with open(label_dst, "w") as f:
                f.write("\n".join(lines))

    print(f"\nDataset created at: {DST_DIR}")
    print(f"Skipped (no image): {skipped_noimg}")

    for phase in ["train", "val", "test"]:
        n_img = len(list((DST_DIR / "images" / phase).iterdir()))
        n_lbl = len(list((DST_DIR / "labels" / phase).iterdir()))
        print(f"  {phase}: {n_img} images, {n_lbl} labels")


if __name__ == "__main__":
    main()