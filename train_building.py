from ultralytics import YOLO

MODEL_WEIGHTS = "yolo11n-seg.pt"
DATA = "/home/fumu/datadisk/building_seg_v5/data.yaml"
PROJECT = "/home/fumu/PycharmProjects/ultralytics-main/runs"
EXPERIMENT_NAME = "building_seg_v5"

EPOCHS = 100
IMGSZ = 640
BATCH = 16
PATIENCE = 50

TRAIN_ARGS = dict(
    data=DATA,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    project=PROJECT,
    name=EXPERIMENT_NAME,
    patience=PATIENCE,
    lr0=0.01,
    overlap_mask=True,
    mask_ratio=4,
)

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"YOLO11n-seg | 5类(building细分) | 标注区域划分 | imgsz={IMGSZ}")
    print(f"epochs={EPOCHS}, batch={BATCH}")
    print(f"{'='*60}\n")

    model = YOLO(MODEL_WEIGHTS)
    model.train(**TRAIN_ARGS)

    print(f"\nbest.pt: {PROJECT}/{EXPERIMENT_NAME}/weights/best.pt")