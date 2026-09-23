import base64
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

SRC_DIRS = [
    Path("/home/fumu/datadisk/标注区域划分/堆龙东嘎"),
    Path("/home/fumu/datadisk/标注区域划分/曲水"),
]
DST_DIR = Path("/home/fumu/datadisk/building_seg_dataset")
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

CLASS_MAP = {
    "residential": 0,
    "steel_roof": 1,
    "large_complex": 2,
    "glass_roof": 3,
    "under_construction": 4,
}


def convert_labelme_to_yolo(json_path):
    d = json.load(open(json_path))
    img_w = d["imageWidth"]
    img_h = d["imageHeight"]
    lines = []
    for shape in d["shapes"]:
        label = shape["label"]
        if label not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[label]
        points = shape["points"]
        coords = []
        for p in points:
            x = max(0.0, min(1.0, p[0] / img_w))
            y = max(0.0, min(1.0, p[1] / img_h))
            coords.append(f"{x:.6f} {y:.6f}")
        if len(coords) >= 3:
            lines.append(f"{cls_id} " + " ".join(coords))
    return lines, d


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


def main():
    random.seed(42)

    for sub in ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"]:
        (DST_DIR / sub).mkdir(parents=True, exist_ok=True)

    all_samples = []
    label_counter = Counter()
    skipped_empty = 0
    skipped_noimg = 0
    CLASS_MAP_REVERSE = {v: k for k, v in CLASS_MAP.items()}

    for src_dir in SRC_DIRS:
        json_files = sorted([f for f in os.listdir(src_dir) if f.endswith(".json")])
        for jf in json_files:
            json_path = src_dir / jf
            lines, d = convert_labelme_to_yolo(json_path)
            if not lines:
                skipped_empty += 1
                continue
            for line in lines:
                cls_id = int(line.split()[0])
                label_counter[CLASS_MAP_REVERSE.get(cls_id, str(cls_id))] += 1
            all_samples.append((json_path, lines, d))

    random.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    splits = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train : n_train + n_val],
        "test": all_samples[n_train + n_val :],
    }

    print(f"\nTotal valid samples: {n}")
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
    print(f"Skipped (empty labels): {skipped_empty}, Skipped (no image): {skipped_noimg}")
    print("\nLabel distribution:")
    for label, count in label_counter.most_common():
        print(f"  {label}: {count}")

    for phase, samples in splits.items():
        img_dir = DST_DIR / "images" / phase
        lbl_dir = DST_DIR / "labels" / phase
        for json_path, lines, d in samples:
            base = json_path.stem
            img_src = find_or_extract_image(json_path, d, img_dir)
            if img_src is None:
                skipped_noimg += 1
                continue
            if img_src.parent == img_dir:
                img_name = img_src.name
            else:
                img_name = img_src.name
                img_dst = img_dir / img_name
                if not img_dst.exists():
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
