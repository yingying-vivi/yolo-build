import logging
import os
import numpy as np
import cv2
import time
from typing import Dict, List, Optional
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YoloSegModel:
    def __init__(self, model_path: str, device: str = None):
        self.model_path = model_path
        self.device = device or ("cuda:0" if self._cuda_available() else "cpu")
        self.yolo_model = None
        self._load_model()

    def _cuda_available(self):
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_model(self):
        logger.info(f"加载 YOLO 分割模型: {self.model_path}")
        logger.info(f"使用设备: {self.device}")
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}，请确保模型权重已部署到正确路径")
        self.yolo_model = YOLO(self.model_path)
        logger.info("YOLO 分割模型加载成功")

    def predict(self, image, conf_threshold: float = 0.5,
                classes: Optional[List[int]] = None) -> Dict:
        results = self.yolo_model(image, conf=conf_threshold, classes=classes, verbose=False)

        boxes_list = []
        masks_list = []
        classes_list = []
        confidences_list = []
        mask_polys_list = []

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for i, box in enumerate(r.boxes):
                cls_id = int(box.cls[0])
                cls_name = self.yolo_model.names[cls_id]
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0])

                mask_np = None
                mask_poly = None
                if r.masks is not None and i < len(r.masks):
                    mask_np = r.masks.data[i].cpu().numpy()
                    try:
                        if hasattr(r.masks, 'xy') and len(r.masks.xy) > i:
                            mask_poly = r.masks.xy[i].copy()
                    except Exception:
                        mask_poly = None

                boxes_list.append([round(v, 1) for v in xyxy])
                masks_list.append(mask_np)
                mask_polys_list.append(mask_poly)
                classes_list.append(cls_name)
                confidences_list.append(round(conf, 3))

        return {
            "boxes": boxes_list,
            "masks": masks_list,
            "mask_polys": mask_polys_list,
            "classes": classes_list,
            "confidences": confidences_list,
            "yolo_names": self.yolo_model.names,
        }


def detect_tiled(model: YoloSegModel, img: np.ndarray,
                 tile_size: int, stride: int, conf_thresh: float) -> List[Dict]:
    h, w = img.shape[:2]
    positions = _tile_positions(h, w, tile_size, stride)
    logger.info(f"图片 {w}x{h}, 切片数 {len(positions)}")

    all_dets = []
    t0 = time.time()

    for idx, (y0, x0) in enumerate(positions):
        tile = img[y0:y0 + tile_size, x0:x0 + tile_size]
        th, tw = tile.shape[:2]
        pad_h = tile_size - th
        pad_w = tile_size - tw
        if pad_h > 0 or pad_w > 0:
            tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

        result = model.predict(tile, conf_threshold=conf_thresh)

        for i in range(len(result["boxes"])):
            bbox_orig = result["boxes"][i]
            x1 = bbox_orig[0] + x0
            y1 = bbox_orig[1] + y0
            x2 = bbox_orig[2] + x0
            y2 = bbox_orig[3] + y0

            if x2 > w + 5 or y2 > h + 5 or x1 < -5 or y1 < -5:
                continue
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))

            bbox = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]

            mask_poly = None
            if result["mask_polys"][i] is not None:
                poly = result["mask_polys"][i].copy()
                poly[:, 0] += x0
                poly[:, 1] += y0
                if pad_h > 0 or pad_w > 0:
                    poly[:, 0] = np.clip(poly[:, 0], 0, w)
                    poly[:, 1] = np.clip(poly[:, 1], 0, h)
                mask_poly = poly

            all_dets.append({
                "class": result["classes"][i],
                "bbox": bbox,
                "conf": result["confidences"][i],
                "mask": result["masks"][i],
                "mask_poly": mask_poly,
            })

        if (idx + 1) % 50 == 0 or idx == len(positions) - 1:
            elapsed = time.time() - t0
            logger.info(f"切片推理进度: {idx + 1}/{len(positions)}, {elapsed:.1f}s")

    logger.info(f"原始检测数: {len(all_dets)}")
    return all_dets


def _tile_positions(h, w, tile_size, stride):
    ys = list(range(0, max(1, h - tile_size + 1), stride))
    xs = list(range(0, max(1, w - tile_size + 1), stride))
    if not ys or ys[-1] + tile_size < h:
        ys.append(max(0, h - tile_size))
    if not xs or xs[-1] + tile_size < w:
        xs.append(max(0, w - tile_size))
    ys = sorted(set(ys))
    xs = sorted(set(xs))
    return [(y, x) for y in ys for x in xs]


def nms(detections: List[Dict], iou_thresh: float) -> List[Dict]:
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: -d["conf"])
    keep = []
    for d in dets:
        suppressed = False
        for k in keep:
            if compute_iou(d["bbox"], k["bbox"]) > iou_thresh:
                suppressed = True
                break
        if not suppressed:
            keep.append(d)
    logger.info(f"NMS: {len(dets)} -> {len(keep)}")
    return keep


def compute_iou(b1, b2):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0