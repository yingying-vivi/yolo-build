import os
from pathlib import Path
from ultralytics import YOLO
import cv2

PROJECT = "/home/fumu/PycharmProjects/ultralytics-main"
IMG_DIR_A = "/home/fumu/PycharmProjects/无人机分割数据集"
IMG_DIR_B = "/home/fumu/PycharmProjects/无人机分割数据集"
OUTPUT_DIR = "/home/fumu/PycharmProjects/ultralytics-main/change_results"
CONF_THRESHOLD = 0.5

MODELS = {
    "YOLOv8": os.path.join(PROJECT, "runs/building_detect_YOLOv8/weights/best.pt"),
    "YOLO26": os.path.join(PROJECT, "runs/building_detect_YOLO26/weights/best.pt"),
}

PAIRS = [
    ("Febnaiqiong0.jpg", "Julynaiqiong0.jpg"),
    ("Febyangda13.jpg", "naiqiongApril13.jpg"),
]

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

def find_new_buildings(boxes_a, boxes_b, iou_threshold=0.3):
    new_buildings = []
    for bb in boxes_b:
        is_new = True
        for ba in boxes_a:
            iou = compute_iou(bb["bbox"], ba["bbox"])
            if iou > iou_threshold:
                is_new = False
                break
        if is_new:
            new_buildings.append(bb)
    return new_buildings

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

def draw_comparison(img_a, img_b, boxes_a, boxes_b, new_buildings, save_path, version):
    h, w = img_a.shape[:2]
    canvas = cv2.hconcat([img_a, img_b])
    cv2.putText(canvas, f"[{version}] Period A", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(canvas, f"[{version}] Period B", (w + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    for ba in boxes_a:
        x1, y1, x2, y2 = [int(v) for v in ba["bbox"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(canvas, f"A:{ba['class']}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for bb in boxes_b:
        x1, y1, x2, y2 = [int(v) for v in bb["bbox"]]
        x1 += w
        x2 += w
        color = (0, 0, 255) if bb in new_buildings else (255, 255, 0)
        label = f"NEW![{version}]" if bb in new_buildings else f"B:{bb['class']}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(save_path, canvas)

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for version, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"[{version}] 模型权重不存在: {model_path}, 请先训练该模型")
            continue

        model = YOLO(model_path)
        print(f"\n{'='*60}")
        print(f"使用模型: {version}  权重: {model_path}")
        print(f"{'='*60}")

        for img_a_name, img_b_name in PAIRS:
            path_a = os.path.join(IMG_DIR_A, img_a_name)
            path_b = os.path.join(IMG_DIR_B, img_b_name)
            boxes_a = detect_buildings(model, path_a)
            boxes_b = detect_buildings(model, path_b)
            new_buildings = find_new_buildings(boxes_a, boxes_b)

            img_a = cv2.imread(path_a)
            img_b = cv2.imread(path_b)
            save_path = os.path.join(OUTPUT_DIR, f"compare_{version}_{img_a_name}_{img_b_name}.png")
            draw_comparison(img_a, img_b, boxes_a, boxes_b, new_buildings, save_path, version)

            print(f"  [{version}] Period A ({img_a_name}): {len(boxes_a)} buildings")
            print(f"  [{version}] Period B ({img_b_name}): {len(boxes_b)} buildings")
            print(f"  [{version}] New buildings: {len(new_buildings)}")
            print(f"  [{version}] Saved: {save_path}")

if __name__ == "__main__":
    main()