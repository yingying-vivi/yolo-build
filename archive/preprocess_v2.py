import base64
import json
import os
import random
import re
import shutil
from collections import Counter
from pathlib import Path

NEW_DATA_DIRS = [
    Path("/home/fumu/datadisk/标注区域划分/堆龙东嘎"),
    Path("/home/fumu/datadisk/标注区域划分/曲水"),
]
OLD_DATA_DIR = Path("/home/fumu/PycharmProjects/无人机分割数据集")
DST_DIR = Path("/home/fumu/datadisk/building_seg_dataset_v2")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

MIN_POINTS = 6

NEW_CLASS_MAP = {
    "residential": "building",
    "steel_roof": "building",
    "large_complex": "building",
    "glass_roof": "building",
    "under_construction": "building",
}
OLD_CLASS_MAP = {
    "building": "building",
}
YOLO_CLASS_ID = {"building": 0}


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


def get_time_group(filename, src_dir_type):
    name = filename.replace(".json", "")
    if src_dir_type == "old":
        m = re.match(r"(dongga|naiqiong|yangda|gurong|niedang)(.*)", name)
        if m:
            return f"old_{m.group(1)}"
        m2 = re.match(r"(Feb|Mar|Apr|May|Jun|July|Jul|Aug|Sep|Oct|Nov|Dec)", name)
        if m2:
            return f"old_{m2.group(1)}"
        return "old_other"
    else:
        return f"new_{Path(str(src_dir_type)).name}"


def main():
    random.seed(42)

    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    for sub in ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"]:
        (DST_DIR / sub).mkdir(parents=True, exist_ok=True)

    all_samples = []
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
                label_counter["building (new)"] += 1
            all_samples.append(("new", json_path, lines, d, str(src_dir)))

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
        all_samples.append(("old", json_path, lines, d, "old"))

    print(f"总标注数: {total_shapes}")
    print(f"过滤(类别不匹配或<6点): {total_filtered} ({total_filtered / total_shapes * 100:.1f}%)")
    print(f"保留标注: {total_shapes - total_filtered}")
    print(f"过滤后空标签样本: {skipped_empty}")

    groups = {}
    for idx, sample in enumerate(all_samples):
        src_type, json_path, lines, d, src_id = sample
        grp = get_time_group(json_path.name, src_id)
        groups.setdefault(grp, []).append(idx)

    train_indices = []
    val_indices = []
    test_indices = []
    for grp, indices in sorted(groups.items()):
        random.shuffle(indices)
        n = len(indices)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        n_test = n - n_train - n_val
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train : n_train + n_val])
        test_indices.extend(indices[n_train + n_val :])
        print(f"  {grp}: total={n}, train={n_train}, val={n_val}, test={n_test}")

    splits = {
        "train": [all_samples[i] for i in train_indices],
        "val": [all_samples[i] for i in val_indices],
        "test": [all_samples[i] for i in test_indices],
    }

    print(f"\nTotal valid samples: {len(all_samples)}")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    print("\nLabel distribution:")
    for label, count in label_counter.most_common():
        print(f"  {label}: {count}")

    skipped_noimg = 0
    for phase, samples in splits.items():
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
