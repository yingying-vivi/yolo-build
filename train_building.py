from ultralytics import YOLO

MODELS = {
    "YOLOv8": {
        "model": "yolov8n.pt",
        "type": "pretrained",
    },
    "YOLO26": {
        "model": "yolo26n.pt",
        "type": "pretrained",
    },
    "YOLOv8Star": {
        "model": "/home/fumu/PycharmProjects/ultralytics-main/ultralytics/cfg/models/v8/yolov8-star.yaml",
        "type": "scratch",
    },
}

DATA = "/home/fumu/PycharmProjects/building_dataset/data.yaml"
PROJECT = "/home/fumu/PycharmProjects/ultralytics-main/runs"
EPOCHS = 200
IMGSZ = 1024
BATCH = 8
PATIENCE = 20

TRAIN_ARGS = dict(
    data=DATA,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    project=PROJECT,
    patience=PATIENCE,
    lr0=0.01,
    augment=True,
    mosaic=1.0,
    mixup=0.1,
    fliplr=0.5,
    flipud=0.5,
    scale=0.5,
)

for version, config in MODELS.items():
    name = f"building_detect_{version}"
    print(f"\n{'=' * 60}")
    print(f"开始训练: {version} (模型: {config['model']}, 类型: {config['type']}, 项目名: {name})")
    print(f"{'=' * 60}\n")
    model = YOLO(config["model"])
    model.train(name=name, **TRAIN_ARGS)
    print(f"\n{version} 训练完成! 结果保存在: {PROJECT}/{name}/")
