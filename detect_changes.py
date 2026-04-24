import os
import re
from pathlib import Path
from ultralytics import YOLO
import cv2

PROJECT = "/home/fumu/PycharmProjects/ultralytics-main"
IMG_DIR = "/home/fumu/PycharmProjects/无人机分割数据集"
BASE_OUTPUT = "/home/fumu/PycharmProjects/ultralytics-main/change_results"
CONF_THRESHOLD = 0.5

MODELS = {
    "YOLOv8": os.path.join(PROJECT, "runs/building_detect_YOLOv8/weights/best.pt"),
    "YOLO26": os.path.join(PROJECT, "runs/building_detect_YOLO26/weights/best.pt"),
    "YOLOv8Star": os.path.join(PROJECT, "runs/building_detect_YOLOv8Star/weights/best.pt"),
    "YOLO11Star": os.path.join(PROJECT, "runs/building_detect_YOLO11Star/weights/best.pt"),
}

PAIRS = [
    ("Febnaiqiong0.jpg", "Julynaiqiong0.jpg"),
    ("Febyangda13.jpg", "naiqiongApril13.jpg"),
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
    boxes = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            boxes.append({
                "class": cls_name,
                "bbox": [round(v, 1) for v in xyxy],
                "conf": round(conf, 3),
            })
    return boxes


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


def find_changes(boxes_a, boxes_b, iou_threshold=0.3, center_dist_threshold=None):
    matched_a = set()
    matched_b = set()
    matches = []

    pairs = []
    for i, ba in enumerate(boxes_a):
        for j, bb in enumerate(boxes_b):
            iou = compute_iou(ba["bbox"], bb["bbox"])
            dist = compute_center_distance(ba["bbox"], bb["bbox"])
            pairs.append((iou, dist, i, j))
    pairs.sort(key=lambda x: -x[0])

    for iou, dist, i, j in pairs:
        if i in matched_a or j in matched_b:
            continue
        if iou < iou_threshold:
            continue
        if center_dist_threshold and dist > center_dist_threshold:
            continue
        matched_a.add(i)
        matched_b.add(j)
        matches.append({
            "box_a": boxes_a[i],
            "box_b": boxes_b[j],
            "iou": round(iou, 3),
            "center_dist": round(dist, 1),
        })

    new_buildings = [boxes_b[j] for j in range(len(boxes_b)) if j not in matched_b]
    disappeared = [boxes_a[i] for i in range(len(boxes_a)) if i not in matched_a]

    return matches, new_buildings, disappeared


def draw_comparison(img_a, img_b, boxes_a, boxes_b, matches, new_buildings, disappeared, save_path, version, time_a, time_b):
    h, w = img_a.shape[:2]
    canvas = cv2.hconcat([img_a, img_b])

    cv2.putText(canvas, f"[{version}] {time_a}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(canvas, f"[{version}] {time_b}", (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    matched_b_indices = set()
    for match in matches:
        matched_b_indices.add(id(match["box_b"]))

    for ba in boxes_a:
        is_disappeared = ba in disappeared
        x1, y1, x2, y2 = [int(v) for v in ba["bbox"]]
        if is_disappeared:
            color = (128, 0, 128)
            label = f"GONE:{ba['class']}"
            thickness = 2
        else:
            color = (0, 255, 0)
            label = f"A:{ba['class']}"
            thickness = 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(canvas, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for bb in boxes_b:
        x1, y1, x2, y2 = [int(v) for v in bb["bbox"]]
        x1 += w
        x2 += w
        if bb in new_buildings:
            color = (0, 0, 255)
            label = f"NEW!:{bb['class']}"
        elif id(bb) in matched_b_indices:
            color = (255, 255, 0)
            label = f"B:{bb['class']}"
        else:
            color = (255, 255, 0)
            label = f"B:{bb['class']}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1, y1 - 5),
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
            boxes_a = detect_buildings(model, path_a)
            boxes_b = detect_buildings(model, path_b)
            matches, new_buildings, disappeared = find_changes(boxes_a, boxes_b)

            img_a = cv2.imread(path_a)
            img_b = cv2.imread(path_b)
            save_path = os.path.join(output_dir, f"{pair_name}.png")
            draw_comparison(img_a, img_b, boxes_a, boxes_b, matches, new_buildings, disappeared,
                            save_path, version, time_a, time_b)

            lines = []
            lines.append(f"\n  [{version}] 对比: {time_a} vs {time_b}")
            lines.append(f"    A期检测: {len(boxes_a)} 建筑物")
            lines.append(f"    B期检测: {len(boxes_b)} 建筑物")
            lines.append(f"    匹配(两期均有): {len(matches)}")
            lines.append(f"    新增: {len(new_buildings)}")
            lines.append(f"    消失(A有B无): {len(disappeared)}")
            for nb in new_buildings:
                lines.append(f"      新增建筑: {nb['class']} conf={nb['conf']} bbox={nb['bbox']}")
            for db in disappeared:
                lines.append(f"      消失建筑: {db['class']} conf={db['conf']} bbox={db['bbox']}")
            for match in matches:
                lines.append(f"      匹配建筑: {match['box_a']['class']} -> {match['box_b']['class']} IOU={match['iou']} 中心距离={match['center_dist']}")
            lines.append(f"    保存: {save_path}")

            for line in lines:
                print(line)
                all_lines.append(line)

    txt_path = os.path.join(BASE_OUTPUT, "detect_report.txt")
    Path(BASE_OUTPUT).mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))
    print(f"\n报告已保存: {txt_path}")


if __name__ == "__main__":
    main()