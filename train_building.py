from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="/home/fumu/PycharmProjects/building_dataset/data.yaml",
    epochs=200,
    imgsz=1024,
    batch=8,
    name="building_detect",
    project="/home/fumu/PycharmProjects/ultralytics-main/runs",
    patience=20,
    lr0=0.01,
    augment=True,
    mosaic=1.0,
    mixup=0.1,
    fliplr=0.5,
    flipud=0.5,
    scale=0.5,
)