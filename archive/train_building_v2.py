from ultralytics import YOLO

MODEL_WEIGHTS = "yolo11n-seg.pt"
DATA = "/home/fumu/datadisk/building_seg_dataset_v3/data.yaml"
PROJECT = "/home/fumu/PycharmProjects/ultralytics-main/runs"
EXPERIMENT_NAME = "building_seg_v3"

EPOCHS = 100
IMGSZ = 640
BATCH = 16
PATIENCE = 50

TRAIN_ARGS = {
    "data": DATA,
    "epochs": EPOCHS,
    "imgsz": IMGSZ,
    "batch": BATCH,
    "project": PROJECT,
    "name": EXPERIMENT_NAME,
    "patience": PATIENCE,
    "lr0": 0.01,
    "overlap_mask": True,
    "mask_ratio": 4,
}

if __name__ == "__main__":
    print(f"\n{'=' * 60}")
    print(f"YOLO11n-seg | 1类(building) | 新旧数据混合 | imgsz={IMGSZ}")
    print(f"epochs={EPOCHS}, batch={BATCH}, patience={PATIENCE}")
    print(f"数据: {DATA}")
    print(f"{'=' * 60}\n")

    model = YOLO(MODEL_WEIGHTS)
    model.train(**TRAIN_ARGS)

    print(f"\n训练完成! best.pt: {PROJECT}/{EXPERIMENT_NAME}/weights/best.pt")
