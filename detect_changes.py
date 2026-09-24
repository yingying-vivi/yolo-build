import os
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

os.environ["YOLO_OFFLINE"] = "True"
os.environ["YOLO_AUTOINSTALL"] = "False"

PROJECT = str(Path(__file__).resolve().parent)
CHANGE_IMG_DIR = "/home/fumu/datadisk/变化"
PREDICT_IMG_DIR = "/home/fumu/datadisk/building_seg_v5_mixed/images/test"
CHANGE_OUTPUT = os.path.join(PROJECT, "change_results/change_detection")
PREDICT_OUTPUT = os.path.join(PROJECT, "change_results/predictions")
CONF_THRESHOLD = 0.5

MODEL_PATH = os.path.join(PROJECT, "runs/building_seg_v5/weights/best.pt")

REGION_CN = {"堆龙": "堆龙东嘎", "曲水": "曲水"}
CLASS_COLORS = {
    "building": (0, 255, 0),
    "car": (255, 255, 0),
    "residential": (0, 200, 0),
    "steel_roof": (0, 180, 0),
    "large_complex": (0, 160, 0),
    "glass_roof": (0, 140, 0),
    "under_construction": (0, 120, 0),
}


def find_change_pairs(img_dir):
    pairs = []
    base = Path(img_dir)
    for region_dir in sorted(base.iterdir()):
        if not region_dir.is_dir():
            continue
        region_cn = REGION_CN.get(region_dir.name, region_dir.name)
        for num_dir in sorted(region_dir.iterdir(), key=lambda d: int(d.name) if d.name.isdigit() else 0):
            if not num_dir.is_dir():
                continue
            imgs = sorted([f for f in num_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
            if len(imgs) >= 2:
                pairs.append((imgs[0], imgs[1], f"{region_cn}_{num_dir.name}"))
    return pairs


def detect_buildings(model, img_path):
    results = model(img_path, conf=CONF_THRESHOLD, verbose=False)
    detections = []
    for r in results:
        for i, box in enumerate(r.boxes):
            mask = None
            if r.masks is not None and i < len(r.masks):
                mask = r.masks.data[i].cpu().numpy()
            detections.append(
                {
                    "class_name": model.names[int(box.cls[0])],
                    "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
                    "conf": round(float(box.conf[0]), 3),
                    "mask": mask,
                }
            )
    return detections


def compute_mask_iou(mask1, mask2, shape):
    if mask1 is None or mask2 is None:
        return 0.0
    m1 = cv2.resize(mask1.astype(np.uint8), (shape[1], shape[0]))
    m2 = cv2.resize(mask2.astype(np.uint8), (shape[1], shape[0]))
    inter = np.logical_and(m1 > 0, m2 > 0).sum()
    union = np.logical_or(m1 > 0, m2 > 0).sum()
    return inter / union if union > 0 else 0.0


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def find_new_buildings(det_a, det_b, img_shape, iou_threshold=0.3):
    matched_a = set()
    matched_b = set()
    pairs = []
    for i, da in enumerate(det_a):
        for j, db in enumerate(det_b):
            bbox_iou = compute_iou(da["bbox"], db["bbox"])
            mask_iou = (
                compute_mask_iou(da["mask"], db["mask"], img_shape)
                if da["mask"] is not None and db["mask"] is not None
                else 0.0
            )
            best_iou = max(bbox_iou, mask_iou)
            pairs.append((best_iou, i, j))
    pairs.sort(key=lambda x: -x[0])
    for best_iou, i, j in pairs:
        if i in matched_a or j in matched_b:
            continue
        if best_iou < iou_threshold:
            continue
        matched_a.add(i)
        matched_b.add(j)
    new_buildings = [det_b[j] for j in range(len(det_b)) if j not in matched_b]
    disappeared = [det_a[i] for i in range(len(det_a)) if i not in matched_a]
    matched = [
        (det_a[i], det_b[j])
        for best_iou, i, j in pairs
        if i in matched_a and j in matched_b and best_iou >= iou_threshold
    ]
    return matched, new_buildings, disappeared


def draw_mask_on_image(canvas, mask, offset_x=0, shape=None, color=(0, 255, 0), alpha=0.4):
    if mask is None or shape is None:
        return
    m = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]))
    overlay = canvas.copy()
    pts = np.where(m > 0)
    for y, x in zip(pts[0], pts[1]):
        x_off = x + offset_x
        if 0 <= x_off < canvas.shape[1]:
            cv2.circle(overlay, (x_off, y), 1, color, -1)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cnt_shifted = cnt + np.array([offset_x, 0])
        cv2.drawContours(canvas, [cnt_shifted], -1, color, 2)


def run_change_detection(model):
    pairs = find_change_pairs(CHANGE_IMG_DIR)
    print(f"[变化检测] 发现 {len(pairs)} 个同期对比对")

    all_lines = []
    for path_a, path_b, tag in pairs:
        img_a = cv2.imread(str(path_a))
        img_b = cv2.imread(str(path_b))
        if img_a is None or img_b is None:
            print(f"  无法读取: {path_a.name} 或 {path_b.name}")
            continue

        building_classes = {
            "residential",
            "steel_roof",
            "large_complex",
            "glass_roof",
            "under_construction",
            "building",
        }
        det_a_all = detect_buildings(model, str(path_a))
        det_b_all = detect_buildings(model, str(path_b))
        det_a = [d for d in det_a_all if d["class_name"] in building_classes]
        det_b = [d for d in det_b_all if d["class_name"] in building_classes]

        img_shape = img_a.shape[:2]
        matched, new_buildings, disappeared = find_new_buildings(det_a, det_b, img_shape)

        h, w = img_a.shape[:2]
        canvas = cv2.hconcat([img_a, img_b])
        cv2.putText(canvas, f"{path_a.name} (前期)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, f"{path_b.name} (后期)", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        matched_b_set = {id(mb) for _, mb in matched}

        for d in det_a_all:
            if d["class_name"] == "car":
                draw_mask_on_image(canvas, d["mask"], offset_x=0, shape=(h, w), color=(255, 255, 0), alpha=0.3)

        for d in det_b_all:
            if d["class_name"] == "car":
                draw_mask_on_image(canvas, d["mask"], offset_x=w, shape=(h, w), color=(255, 255, 0), alpha=0.3)

        for da in det_a:
            if da in disappeared:
                draw_mask_on_image(canvas, da["mask"], offset_x=0, shape=(h, w), color=(128, 0, 128))
                x1, y1, x2, y2 = [int(v) for v in da["bbox"]]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (128, 0, 128), 2)
            else:
                draw_mask_on_image(canvas, da["mask"], offset_x=0, shape=(h, w), color=(0, 200, 0))
                x1, y1, x2, y2 = [int(v) for v in da["bbox"]]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 1)

        for db in det_b:
            if db in new_buildings:
                draw_mask_on_image(canvas, db["mask"], offset_x=w, shape=(h, w), color=(0, 0, 255), alpha=0.5)
                x1, y1, x2, y2 = [int(v) for v in db["bbox"]]
                cv2.rectangle(canvas, (x1 + w, y1), (x2 + w, y2), (0, 0, 255), 3)
                cv2.putText(canvas, "NEW", (x1 + w, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif id(db) in matched_b_set:
                draw_mask_on_image(canvas, db["mask"], offset_x=w, shape=(h, w), color=(0, 200, 0))
                x1, y1, x2, y2 = [int(v) for v in db["bbox"]]
                cv2.rectangle(canvas, (x1 + w, y1), (x2 + w, y2), (0, 200, 0), 1)
            else:
                draw_mask_on_image(canvas, db["mask"], offset_x=w, shape=(h, w), color=(0, 200, 0))

        legend_y = h - 20
        cv2.putText(
            canvas,
            "Green=Existing  Red=NEW  Purple=Disappeared  Yellow=Car",
            (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        save_path = os.path.join(CHANGE_OUTPUT, f"{tag}.png")
        cv2.imwrite(save_path, canvas)

        lines = [
            f"{tag}: {path_a.name} vs {path_b.name} | 建筑{len(det_a)} vs {len(det_b)} | 匹配{len(matched)} | 新增{len(new_buildings)} | 消失{len(disappeared)}"
        ]
        for nb in new_buildings:
            lines.append(f"  新增: conf={nb['conf']} bbox={nb['bbox']}")
        for line in lines:
            print(line)
            all_lines.append(line)

    report_path = os.path.join(CHANGE_OUTPUT, "detect_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))
    print(f"\n报告: {report_path}")


def run_prediction(model):
    img_files = sorted([f for f in os.listdir(PREDICT_IMG_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    print(f"\n[单图检测] 检测 {len(img_files)} 张test图片")

    all_lines = []
    for img_name in img_files:
        img_path = os.path.join(PREDICT_IMG_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        det = detect_buildings(model, img_path)

        canvas = img.copy()
        for d in det:
            color = CLASS_COLORS.get(d["class_name"], (0, 255, 0))
            draw_mask_on_image(canvas, d["mask"], shape=(h, w), color=color)
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                canvas, f"{d['class_name']} {d['conf']:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        save_path = os.path.join(PREDICT_OUTPUT, img_name)
        cv2.imwrite(save_path, canvas)

        line = f"{img_name}: {len(det)} targets"
        print(line)
        all_lines.append(line)

    report_path = os.path.join(PREDICT_OUTPUT, "predict_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"模型不存在: {MODEL_PATH}")
        print("请先运行 train_building_mixed.py")
        return

    model = YOLO(MODEL_PATH)

    Path(CHANGE_OUTPUT).mkdir(parents=True, exist_ok=True)
    Path(PREDICT_OUTPUT).mkdir(parents=True, exist_ok=True)

    run_change_detection(model)
    run_prediction(model)


if __name__ == "__main__":
    main()
