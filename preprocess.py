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
DST_DIR = Path("/home/fumu/datadisk/building_seg_v5")

MIN_POINTS = 4

CLASS_MAP = {
    "residential": 0,
    "steel_roof": 1,
    "large_complex": 2,
    "glass_roof": 3,
    "under_construction": 4,
}


def convert_labelme_to_yolo(json_path):
    d = json.load(open(json_path, "r"))
    img_w = d["imageWidth"]
    img_h = d["imageHeight"]
    lines = []
    filtered = 0
    total = 0
    for shape in d["shapes"]:
        total += 1
        label = shape["label"]
        cls_id = CLASS_MAP.get(label)
        if cls_id is None:
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
            lines, d, total, filtered = convert_labelme_to_yolo(json_path)
            total_shapes += total
            total_filtered += filtered
            if not lines:
                skipped_empty += 1
                continue
            for line in lines:
                cls_id = int(line.split()[0])
                label_counter[cls_id] += 1
            idx = len(all_samples)
            all_samples.append((json_path, lines, d, str(src_dir)))

    CLASS_NAMES = {0: "residential", 1: "steel_roof", 2: "large_complex", 3: "glass_roof", 4: "under_construction"}
    print(f"总标注: {total_shapes}")
    print(f"过滤(<{MIN_POINTS}点): {total_filtered} ({total_filtered/total_shapes*100:.1f}%)")
    print(f"保留标注: {total_shapes - total_filtered}")
    print(f"空标签样本: {skipped_empty}")
    print(f"\n保留标注类别:")
    for cls_id, count in label_counter.most_common():
        print(f"  {CLASS_NAMES[cls_id]}: {count}")

    # Simple random split
    random.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    splits = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train:n_train + n_val],
        "test": all_samples[n_train + n_val:],
    }

    print(f"\nTrain: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    skipped_noimg = 0
    for phase, samples in splits.items():
        img_dir = DST_DIR / "images" / phase
        lbl_dir = DST_DIR / "labels" / phase
        for json_path, lines, d, src_id in samples:
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