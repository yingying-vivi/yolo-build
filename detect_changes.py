import os
import re
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

PROJECT = "/home/fumu/PycharmProjects/ultralytics-main"
IMG_DIR = "/home/fumu/datasets/无人机分割数据集"
BASE_OUTPUT = "/home/fumu/PycharmProjects/ultralytics-main/change_results"
CONF_THRESHOLD = 0.5

ACTIVE_MODEL = "YOLO11n-seg"

ALL_MODELS = {
    "YOLO11n-seg": os.path.join(PROJECT, "runs/segment/train-3/weights/best.pt"),
}

MODELS = {ACTIVE_MODEL: ALL_MODELS[ACTIVE_MODEL]}

MONTH_ORDER = ["Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

PAIRS = [
    ("Febnaiqiong11.jpg", "Julynaiqiong11.jpg"),
    ("Febnaiqiong10.jpg", "Julynaiqiong10.jpg"),
    ("Febnaiqiong0.jpg", "Julynaiqiong0.jpg"),
    ("Febnaiqiong1.jpg", "Julynaiqiong1.jpg"),
    ("Febnaiqiong8.jpg", "Julynaiqiong8.jpg"),
]

MONTH_CN = {
    "January": "1月", "Jan": "1月",
    "February": "2月", "Feb": "2月",
    "March": "3月", "Mar": "3月",
    "April": "4月", "Apr": "4月",
    "May": "5月",
    "June": "6月", "Jun": "6月",
    "July": "7月", "Jul": "7月",
    "August": "8月", "Aug": "8月",
    "September": "9月", "Sep": "9月",
    "October": "10月", "Oct": "10月",
    "November": "11月", "Nov": "11月",
    "December": "12月", "Dec": "12月",
}

LOCATION_CN = {
    "naiqiong": "乃琼", "yangda": "羊达", "dongga": "东嘎",
    "gurong": "古荣", "niedang": "尼达", "qushui": "曲水",
    "deqing": "德庆",
}

_LOC_PATTERN = "|".join(sorted(LOCATION_CN.keys(), key=len, reverse=True))
_MONTH_PATTERN = "|".join(sorted(MONTH_CN.keys(), key=len, reverse=True))


def extract_time_tag(filename):
    name = filename.rsplit(".", 1)[0]
    m = re.match(rf"({_MONTH_PATTERN})({_LOC_PATTERN})(.*)", name)
    if m:
        month = MONTH_CN.get(m.group(1), m.group(1))
        loc = LOCATION_CN.get(m.group(2), m.group(2))
        num = m.group(3)
        return f"{month}-{loc}-{num}"
    m2 = re.match(rf"({_LOC_PATTERN})({_MONTH_PATTERN})(.*)", name)
    if m2:
        loc = LOCATION_CN.get(m2.group(1), m2.group(1))
        month = MONTH_CN.get(m2.group(2), m2.group(2))
        num = m2.group(3)
        return f"{month}-{loc}-{num}"
    return name


def detect_buildings(model, img_path):
    results = model(img_path, conf=CONF_THRESHOLD, verbose=False)
    detections = []
    for r in results:
        for i, box in enumerate(r.boxes):
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            mask = None
            if r.masks is not None and i < len(r.masks):
                mask_np = r.masks.data[i].cpu().numpy()
                mask = mask_np
            detections.append({
                "class": cls_name,
                "bbox": [round(v, 1) for v in xyxy],
                "conf": round(conf, 3),
                "mask": mask,
            })
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


def compute_center_distance(box1, box2):
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


def find_changes(det_a, det_b, img_shape, iou_threshold=0.3, mask_iou_threshold=0.2):
    matched_a = set()
    matched_b = set()
    matches = []

    pairs = []
    for i, da in enumerate(det_a):
        for j, db in enumerate(det_b):
            bbox_iou = compute_iou(da["bbox"], db["bbox"])
            mask_iou = compute_mask_iou(da["mask"], db["mask"], img_shape) if da["mask"] is not None and db["mask"] is not None else 0.0
            best_iou = max(bbox_iou, mask_iou)
            dist = compute_center_distance(da["bbox"], db["bbox"])
            pairs.append((best_iou, dist, i, j))
    pairs.sort(key=lambda x: -x[0])

    for best_iou, dist, i, j in pairs:
        if i in matched_a or j in matched_b:
            continue
        if best_iou < iou_threshold:
            continue
        matched_a.add(i)
        matched_b.add(j)
        mask_iou_val = compute_mask_iou(det_a[i]["mask"], det_b[j]["mask"], img_shape) if det_a[i]["mask"] is not None and det_b[j]["mask"] is not None else 0.0
        matches.append({
            "box_a": det_a[i],
            "box_b": det_b[j],
            "bbox_iou": round(compute_iou(det_a[i]["bbox"], det_b[j]["bbox"]), 3),
            "mask_iou": round(mask_iou_val, 3),
            "center_dist": round(dist, 1),
        })

    new_buildings = [det_b[j] for j in range(len(det_b)) if j not in matched_b]
    disappeared = [det_a[i] for i in range(len(det_a)) if i not in matched_a]

    return matches, new_buildings, disappeared


def draw_mask_on_image(canvas, mask, offset_x=0, shape=None, color=(0, 255, 0), alpha=0.35):
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


def draw_comparison(img_a, img_b, det_a, det_b, matches, new_buildings, disappeared,
                    save_path, version, time_a, time_b):
    h, w = img_a.shape[:2]
    canvas = cv2.hconcat([img_a, img_b])

    cv2.putText(canvas, f"[{version}] {time_a}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(canvas, f"[{version}] {time_b}", (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    matched_b_indices = set()
    for match in matches:
        matched_b_indices.add(id(match["box_b"]))

    for da in det_a:
        is_disappeared = da in disappeared
        if is_disappeared:
            color = (128, 0, 128)
            label = f"GONE:{da['class']}"
        else:
            color = (0, 255, 0)
            label = f"A:{da['class']}"
        draw_mask_on_image(canvas, da["mask"], offset_x=0, shape=(h, w), color=color)
        x1, y1, x2, y2 = [int(v) for v in da["bbox"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for db in det_b:
        if db in new_buildings:
            color = (0, 0, 255)
            label = f"NEW!:{db['class']}"
        elif id(db) in matched_b_indices:
            color = (255, 255, 0)
            label = f"B:{db['class']}"
        else:
            color = (255, 255, 0)
            label = f"B:{db['class']}"
        draw_mask_on_image(canvas, db["mask"], offset_x=w, shape=(h, w), color=color)
        x1, y1, x2, y2 = [int(v) for v in db["bbox"]]
        cv2.rectangle(canvas, (x1 + w, y1), (x2 + w, y2), color, 2)
        cv2.putText(canvas, label, (x1 + w, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    legend_y = h - 20
    cv2.putText(canvas, "Green=A period  Yellow=Still exists  Red=NEW!  Purple=GONE(A only)",
                (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imwrite(save_path, canvas)


def main():
    all_lines = []

    for version, model_path in MODELS.items():
        if not os.path.exists(model_path):
            line = f"[{version}] 模型权重不存在: {model_path}, 请先训练该模型"
            print(line)
            all_lines.append(line)
            continue

        output_dir = os.path.join(BASE_OUTPUT, version)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        model = YOLO(model_path)
        header = f"\n{'='*60}\n使用模型: {version}  权重: {model_path}\n输出目录: {output_dir}\n{'='*60}"
        print(header)
        all_lines.append(header)

        for img_a_name, img_b_name in PAIRS:
            time_a = extract_time_tag(img_a_name)
            time_b = extract_time_tag(img_b_name)
            pair_name = f"{time_a}--{time_b}"

            path_a = os.path.join(IMG_DIR, img_a_name)
            path_b = os.path.join(IMG_DIR, img_b_name)
            img_a = cv2.imread(path_a)
            img_b = cv2.imread(path_b)
            if img_a is None or img_b is None:
                line = f"  无法读取图片: {img_a_name} 或 {img_b_name}"
                print(line)
                all_lines.append(line)
                continue

            det_a = detect_buildings(model, path_a)
            det_b = detect_buildings(model, path_b)
            img_shape = img_a.shape[:2]
            matches, new_buildings, disappeared = find_changes(det_a, det_b, img_shape)

            save_path = os.path.join(output_dir, f"{pair_name}.png")
            draw_comparison(img_a, img_b, det_a, det_b, matches, new_buildings, disappeared,
                            save_path, version, time_a, time_b)

            lines = []
            lines.append(f"\n  [{version}] 对比: {time_a} vs {time_b}")
            lines.append(f"    A期检测: {len(det_a)} 建筑物")
            lines.append(f"    B期检测: {len(det_b)} 建筑物")
            lines.append(f"    匹配(两期均有): {len(matches)}")
            lines.append(f"    新增: {len(new_buildings)}")
            lines.append(f"    消失(A有B无): {len(disappeared)}")
            for nb in new_buildings:
                lines.append(f"      新增建筑: {nb['class']} conf={nb['conf']} bbox={nb['bbox']}")
            for db in disappeared:
                lines.append(f"      消失建筑: {db['class']} conf={db['conf']} bbox={db['bbox']}")
            for match in matches:
                lines.append(f"      匹配建筑: {match['box_a']['class']} -> {match['box_b']['class']} bbox_iou={match['bbox_iou']} mask_iou={match['mask_iou']} 中心距离={match['center_dist']}")
            lines.append(f"    保存: {save_path}")

            for line in lines:
                print(line)
                all_lines.append(line)

    txt_path = os.path.join(BASE_OUTPUT, ACTIVE_MODEL, "detect_report.txt")
    Path(os.path.join(BASE_OUTPUT, ACTIVE_MODEL)).mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))
    print(f"\n报告已保存: {txt_path}")


if __name__ == "__main__":
    main()