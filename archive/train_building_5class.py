from ultralytics import YOLO

MODEL_WEIGHTS = "yolo11n-seg.pt"
DATA = "/home/fumu/datadisk/building_seg_dataset/data.yaml"
PROJECT = "/home/fumu/PycharmProjects/ultralytics-main/runs"
EXPERIMENT_NAME = "building_seg_5class"

EPOCHS = 200
IMGSZ = 1024
BATCH = 8
PATIENCE = 30

TRAIN_ARGS = dict(
    data=DATA,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    project=PROJECT,
    name=EXPERIMENT_NAME,
    patience=PATIENCE,
    lr0=0.01,
    augment=True,
    mosaic=1.0,
    mixup=0.1,
    fliplr=0.5,
    flipud=0.5,
    scale=0.5,
    overlap_mask=True,
    mask_ratio=4,
)

if __name__ == "__main__":
    print(f"\n{'=' * 60}")
    print(f"训练 YOLO11n-seg | 数据: {DATA} | 类别数: 5")
    print(f"epochs={EPOCHS}, imgsz={IMGSZ}, batch={BATCH}, patience={PATIENCE}")
    print(f"结果保存: {PROJECT}/{EXPERIMENT_NAME}")
    print(f"{'=' * 60}\n")

    model = YOLO(MODEL_WEIGHTS)
    model.train(**TRAIN_ARGS)

    print(f"\n训练完成! best.pt: {PROJECT}/{EXPERIMENT_NAME}/weights/best.pt")
    print(f"last.pt: {PROJECT}/{EXPERIMENT_NAME}/weights/last.pt")
