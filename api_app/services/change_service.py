import os
import json
import logging
import time
import numpy as np
import cv2
from pathlib import Path

from ..models.model_wrapper import YoloSegModel, detect_tiled, nms, compute_iou
from ..models.model_config_loader import ModelConfigLoader
from ..utils.image_utils import (
    get_overlap_bounds, warp_to_overlap, read_as_bgr_uint8,
    crop_black_border, write_geotiff, write_changes_to_shapefile, HAS_GDAL,
)
from ..utils.file_utils import zip_shapefile

if HAS_GDAL:
    from osgeo import gdal

logger = logging.getLogger(__name__)


class ChangeDetectionService:

    def run_change_detection(
        self,
        t1_path: str,
        t2_path: str,
        model_path: str = None,
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
        match_iou_threshold: float = 0.3,
        area_change_threshold: float = 0.2,
        tile_size: int = 1024,
        stride: int = 800,
        target_res: float = 0.2,
        vis_tile_size: int = 1024,
        output_dir: str = None,
        task_id: str = None,
    ) -> dict:
        t_start = time.time()

        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), "..", "api_results")
        if task_id is None:
            import uuid
            task_id = uuid.uuid4().hex

        task_dir = os.path.join(output_dir, task_id)
        Path(task_dir).mkdir(parents=True, exist_ok=True)

        if model_path is None:
            config_loader = ModelConfigLoader()
            model_path = config_loader.get_yolo_seg_model_path()
            logger.info(f"使用默认模型: {model_path}")

        logger.info("=" * 60)
        logger.info("建筑物变化检测")
        logger.info(f"基期: {t1_path}")
        logger.info(f"检测期: {t2_path}")
        logger.info(f"模型: {model_path}")
        logger.info(f"输出: {task_dir}")
        logger.info(f"置信度={confidence_threshold}, NMS IoU={nms_iou_threshold}, "
                    f"匹配IoU={match_iou_threshold}, 面积变化阈值={area_change_threshold}")

        warp_res = target_res
        bounds = None
        arr1 = None
        gt_out = None
        crs_out = None

        if HAS_GDAL:
            logger.info("[1] 计算重叠区域...")
            try:
                bounds = get_overlap_bounds(t1_path, t2_path)
            except (ValueError, RuntimeError) as e:
                logger.warning(f"重叠区域计算失败({e}), 将直接读取影像")

            try:
                ds1_info = gdal.Open(t1_path)
                gt1_src = ds1_info.GetGeoTransform()
                ds1_info = None
                pixel_size = abs(gt1_src[1])
                if pixel_size < 0.01 and target_res > 0.01:
                    logger.info(f"影像为经纬度坐标(像素={pixel_size}), target_res={target_res}度过大, 自动调整为源影像分辨率")
                    warp_res = pixel_size
                else:
                    logger.info(f"影像为投影坐标(像素={pixel_size}), 使用target_res={target_res}")
            except Exception as e:
                logger.warning(f"读取影像GeoTransform失败({e})")

            if bounds:
                try:
                    logger.info("[2] GDAL配准+裁剪...")
                    p1_warped = os.path.join(task_dir, "period1_aligned.tif")
                    p2_warped = os.path.join(task_dir, "period2_aligned.tif")
                    warp_to_overlap(t1_path, bounds, warp_res, p1_warped)
                    warp_to_overlap(t2_path, bounds, warp_res, p2_warped)

                    logger.info("[3] 读取为BGR uint8...")
                    arr1, gt_out, crs_out = read_as_bgr_uint8(p1_warped)
                    arr2, gt2, crs2 = read_as_bgr_uint8(p2_warped)
                    h = min(arr1.shape[0], arr2.shape[0])
                    w = min(arr1.shape[1], arr2.shape[1])
                    arr1 = arr1[:h, :w]
                    arr2 = arr2[:h, :w]

                    logger.info("[3.5] 去除黑边...")
                    arr1, arr2, gt_out = crop_black_border(arr1, arr2, gt_out)
                except Exception as e:
                    logger.warning(f"GDAL配准裁剪失败({e}), 回退到直接读取影像")
                    bounds = None

        if arr1 is None:
            logger.info("直接读取影像（不经过GDAL配准）...")
            arr1, gt_out, crs_out = read_as_bgr_uint8(t1_path)
            arr2, gt2, crs2 = read_as_bgr_uint8(t2_path)
            bounds = None

        logger.info(f"影像尺寸: T1={arr1.shape}, T2={arr2.shape}")

        logger.info("[4] 加载YOLO模型...")
        model = YoloSegModel(model_path)

        logger.info("[5] 第一期(基期)切片推理...")
        raw_dets1 = detect_tiled(model, arr1, tile_size, stride, confidence_threshold)
        det1 = nms(raw_dets1, nms_iou_threshold)

        logger.info("[6] 第二期(检测期)切片推理...")
        raw_dets2 = detect_tiled(model, arr2, tile_size, stride, confidence_threshold)
        det2 = nms(raw_dets2, nms_iou_threshold)

        logger.info("[7] 变化检测匹配...")
        matches, new_buildings, disappeared = self._find_changes(
            det1, det2, match_iou_threshold
        )

        logger.info(f"基期建筑物: {len(det1)}")
        logger.info(f"检测期建筑物: {len(det2)}")
        logger.info(f"匹配(两期均有): {len(matches)}")
        logger.info(f"新增建筑物: {len(new_buildings)}")
        logger.info(f"消失建筑物: {len(disappeared)}")

        changes_dict = self._classify_changes(
            matches, new_buildings, disappeared, det1, det2,
            area_change_threshold
        )

        logger.info("[8] 保存变化结果JSON...")
        json_path = os.path.join(task_dir, f"instance_changes_{task_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(changes_dict, f, ensure_ascii=False, indent=2)

        logger.info("[9] 生成切片对比图...")
        self._create_comparison_tiles(
            arr1, arr2, det1, matches, new_buildings, disappeared,
            task_dir, vis_tile_size
        )

        logger.info("[10] 生成总览图...")
        self._create_overview(
            arr1, arr2, det1, matches, new_buildings, disappeared,
            task_dir
        )

        logger.info("[11] 保存新增建筑物掩膜GeoTIFF...")
        try:
            self._save_new_buildings_mask(
                new_buildings, arr2, gt_out, crs_out, task_dir
            )
        except Exception as e:
            logger.warning(f"保存掩膜GeoTIFF失败(不影响主流程): {e}")

        logger.info("[12] 生成Shapefile...")
        shp_dir = None
        zip_path = None
        if HAS_GDAL and gt_out is not None:
            try:
                shp_dir = write_changes_to_shapefile(
                    changes_dict, gt_out, crs_out, task_dir, task_id
                )
                if shp_dir:
                    zip_path = zip_shapefile(
                        shp_dir, os.path.join(task_dir, f"instance_changes_{task_id}.zip")
                    )
            except Exception as e:
                logger.warning(f"生成Shapefile失败(不影响主流程): {e}")

        logger.info("[13] 生成报告...")
        report = self._generate_report(
            t1_path, t2_path, model_path, confidence_threshold,
            nms_iou_threshold, match_iou_threshold, area_change_threshold,
            tile_size, stride, arr1, det1, det2, matches,
            new_buildings, disappeared, bounds, task_id, t_start
        )
        report_path = os.path.join(task_dir, "detect_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        elapsed = time.time() - t_start
        logger.info(f"变化检测完成，总耗时: {elapsed:.1f}s")
        logger.info(f"结果目录: {task_dir}")

        return {
            "changes_dict": changes_dict,
            "json_path": json_path,
            "zip_path": zip_path,
            "report_path": report_path,
            "task_dir": task_dir,
        }

    def _find_changes(self, det_a, det_b, iou_thresh=0.3):
        matched_a = set()
        matched_b = set()
        matches = []

        pairs = []
        for i, da in enumerate(det_a):
            for j, db in enumerate(det_b):
                iou_val = compute_iou(da["bbox"], db["bbox"])
                pairs.append((iou_val, i, j))
        pairs.sort(key=lambda x: -x[0])

        for iou_val, i, j in pairs:
            if i in matched_a or j in matched_b:
                continue
            if iou_val < iou_thresh:
                continue
            matched_a.add(i)
            matched_b.add(j)
            matches.append({"box_a": det_a[i], "box_b": det_b[j], "iou": round(iou_val, 3)})

        new_buildings = [det_b[j] for j in range(len(det_b)) if j not in matched_b]
        disappeared = [det_a[i] for i in range(len(det_a)) if i not in matched_a]
        return matches, new_buildings, disappeared

    def _classify_changes(self, matches, new_buildings, disappeared,
                          det1, det2, area_change_threshold=0.2):
        changes = []

        for nb in new_buildings:
            changes.append({
                "type": "new",
                "bbox": nb["bbox"],
                "confidence": nb.get("conf", 1.0),
                "class": nb.get("class", "Building"),
            })

        for da in disappeared:
            changes.append({
                "type": "removed",
                "bbox": da["bbox"],
                "confidence": da.get("conf", 1.0),
                "class": da.get("class", "Building"),
            })

        for match in matches:
            box_a = match["box_a"]
            box_b = match["box_b"]
            area_a = (box_a["bbox"][2] - box_a["bbox"][0]) * (box_a["bbox"][3] - box_a["bbox"][1])
            area_b = (box_b["bbox"][2] - box_b["bbox"][0]) * (box_b["bbox"][3] - box_b["bbox"][1])
            area_change_ratio = abs(area_b - area_a) / area_a if area_a > 0 else 0.0

            if area_change_ratio > area_change_threshold:
                changes.append({
                    "type": "expanded",
                    "bbox": box_b["bbox"],
                    "confidence": box_b.get("conf", 1.0),
                    "area_t1": round(area_a, 1),
                    "area_t2": round(area_b, 1),
                    "area_change_ratio": round(area_change_ratio, 3),
                    "class": box_b.get("class", "Building"),
                })

        result = {
            "total_changes": len(changes),
            "new_count": len(new_buildings),
            "removed_count": len(disappeared),
            "expanded_count": len([c for c in changes if c["type"] == "expanded"]),
            "matched_count": len(matches),
            "t1_detection_count": len(det1),
            "t2_detection_count": len(det2),
            "changes": changes,
        }
        logger.info(f"变化分类完成: 新建{result['new_count']}, "
                    f"移除{result['removed_count']}, 扩张{result['expanded_count']}")
        return result

    def _create_comparison_tiles(self, arr1, arr2, det1, matches,
                                  new_buildings, disappeared,
                                  output_dir, vis_tile_size):
        h, w = arr1.shape[:2]
        tile_dir = os.path.join(output_dir, "comparison_tiles")
        Path(tile_dir).mkdir(parents=True, exist_ok=True)

        matched_b_ids = {id(m["box_b"]) for m in matches}
        disappeared_ids = {id(d) for d in disappeared}
        new_ids = {id(nb) for nb in new_buildings}

        n_rows = h // vis_tile_size
        n_cols = w // vis_tile_size
        if n_rows == 0 or n_cols == 0:
            logger.warning(f"影像太小({w}x{h})，无法生成切片对比图")
            return tile_dir, 0, 0

        positions = []
        for row_idx, y in enumerate(range(0, n_rows * vis_tile_size, vis_tile_size)):
            for col_idx, x in enumerate(range(0, n_cols * vis_tile_size, vis_tile_size)):
                positions.append((y, x, row_idx, col_idx))

        has_change_tiles = 0
        total = len(positions)

        for idx, (y0, x0, row_idx, col_idx) in enumerate(positions):
            crop1 = arr1[y0:y0 + vis_tile_size, x0:x0 + vis_tile_size]
            crop2 = arr2[y0:y0 + vis_tile_size, x0:x0 + vis_tile_size]
            tile_bbox = [x0, y0, x0 + vis_tile_size, y0 + vis_tile_size]

            canvas_a = crop1.copy()
            canvas_b = crop2.copy()
            tile_has_change = False

            for da in det1:
                if not _bbox_overlap(da["bbox"], tile_bbox):
                    continue
                is_disappeared = id(da) in disappeared_ids
                color = (128, 0, 128) if is_disappeared else (0, 200, 0)
                label = "GONE" if is_disappeared else ""
                poly = _shift_poly_to_tile(da.get("mask_poly"), x0, y0)
                _draw_mask_poly_on_tile(canvas_a, poly, color, alpha=0.3 if is_disappeared else 0.2)
                x1t = max(0, int(da["bbox"][0]) - x0)
                y1t = max(0, int(da["bbox"][1]) - y0)
                x2t = min(vis_tile_size, int(da["bbox"][2]) - x0)
                y2t = min(vis_tile_size, int(da["bbox"][3]) - y0)
                cv2.rectangle(canvas_a, (x1t, y1t), (x2t, y2t), color, 2)
                if label:
                    cv2.putText(canvas_a, label, (x1t, max(12, y1t - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                if is_disappeared:
                    tile_has_change = True

            for nb in new_buildings:
                if not _bbox_overlap(nb["bbox"], tile_bbox):
                    continue
                color = (0, 0, 255)
                poly = _shift_poly_to_tile(nb.get("mask_poly"), x0, y0)
                _draw_mask_poly_on_tile(canvas_b, poly, color, alpha=0.5)
                x1t = max(0, int(nb["bbox"][0]) - x0)
                y1t = max(0, int(nb["bbox"][1]) - y0)
                x2t = min(vis_tile_size, int(nb["bbox"][2]) - x0)
                y2t = min(vis_tile_size, int(nb["bbox"][3]) - y0)
                cv2.rectangle(canvas_b, (x1t, y1t), (x2t, y2t), color, 3)
                cv2.putText(canvas_b, "NEW!", (x1t, max(12, y1t - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                tile_has_change = True

            for m in matches:
                db = m["box_b"]
                if not _bbox_overlap(db["bbox"], tile_bbox):
                    continue
                color = (0, 200, 0)
                poly = _shift_poly_to_tile(db.get("mask_poly"), x0, y0)
                _draw_mask_poly_on_tile(canvas_b, poly, color, alpha=0.2)
                x1t = max(0, int(db["bbox"][0]) - x0)
                y1t = max(0, int(db["bbox"][1]) - y0)
                x2t = min(vis_tile_size, int(db["bbox"][2]) - x0)
                y2t = min(vis_tile_size, int(db["bbox"][3]) - y0)
                cv2.rectangle(canvas_b, (x1t, y1t), (x2t, y2t), color, 2)

            comparison = cv2.hconcat([canvas_a, canvas_b])
            cv2.line(comparison, (vis_tile_size, 0), (vis_tile_size, vis_tile_size),
                     (255, 255, 255), 2)
            cv2.putText(comparison, "基期(第一期)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(comparison, "检测期(第二期)", (vis_tile_size + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            change_tag = "_change" if tile_has_change else ""
            tile_path = os.path.join(tile_dir, f"tile_r{row_idx}_c{col_idx}{change_tag}.png")
            cv2.imwrite(tile_path, comparison)

            if tile_has_change:
                has_change_tiles += 1

        logger.info(f"对比切片生成完成: 有变化={has_change_tiles}/{total}")
        return tile_dir, has_change_tiles, total

    def _create_overview(self, arr1, arr2, det1, matches,
                          new_buildings, disappeared, output_dir, scale=0.1):
        h, w = arr1.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        if new_w < 10 or new_h < 10:
            logger.warning("影像太小，无法生成总览图")
            return None

        thumb1 = cv2.resize(arr1, (new_w, new_h), interpolation=cv2.INTER_AREA)
        thumb2 = cv2.resize(arr2, (new_w, new_h), interpolation=cv2.INTER_AREA)

        for nb in new_buildings:
            x1, y1, x2, y2 = [int(v * scale) for v in nb["bbox"]]
            cv2.rectangle(thumb2, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(thumb2, "NEW", (x1, max(8, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        for da in disappeared:
            x1, y1, x2, y2 = [int(v * scale) for v in da["bbox"]]
            cv2.rectangle(thumb1, (x1, y1), (x2, y2), (128, 0, 128), 2)

        for m in matches:
            db = m["box_b"]
            x1, y1, x2, y2 = [int(v * scale) for v in db["bbox"]]
            cv2.rectangle(thumb2, (x1, y1), (x2, y2), (0, 200, 0), 1)

        overview = cv2.hconcat([thumb1, thumb2])
        cv2.line(overview, (new_w, 0), (new_w, new_h), (255, 255, 255), 3)
        cv2.putText(overview, "基期(第一期)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overview, "检测期(第二期)", (new_w + 10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overview, "Red=NEW! Purple=GONE Green=Existing",
                    (10, new_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        overview_path = os.path.join(output_dir, "overview_comparison.png")
        cv2.imwrite(overview_path, overview)
        logger.info(f"总览保存: {overview_path}")
        return overview_path

    def _save_new_buildings_mask(self, new_buildings, arr2, gt, crs, task_dir):
        new_mask = np.zeros(arr2.shape[:2], dtype=np.uint8)
        for nb in new_buildings:
            if nb.get("mask_poly") is not None and len(nb["mask_poly"]) >= 3:
                cv2.fillPoly(new_mask, [nb["mask_poly"].astype(np.int32)], 255)
            else:
                x1, y1, x2, y2 = [int(v) for v in nb["bbox"]]
                new_mask[max(0, y1):y2, max(0, x1):x2] = 255

        mask_path = os.path.join(task_dir, "new_buildings_mask.tif")
        if HAS_GDAL:
            write_geotiff(mask_path, new_mask, gt, crs)
        else:
            cv2.imwrite(mask_path, new_mask)

    def _generate_report(self, t1_path, t2_path, model_path, conf_thresh,
                          nms_iou, match_iou, area_thresh, tile_size, stride,
                          arr1, det1, det2, matches, new_buildings, disappeared,
                          bounds, task_id, t_start):
        elapsed = time.time() - t_start
        lines = []
        lines.append("=" * 60)
        lines.append("建筑物变化检测报告")
        lines.append("=" * 60)
        lines.append(f"基期(第一期): {t1_path}")
        lines.append(f"检测期(第二期): {t2_path}")
        lines.append(f"模型: {model_path}")
        lines.append(f"置信度阈值: {conf_thresh}")
        lines.append(f"NMS IoU阈值: {nms_iou}")
        lines.append(f"匹配IoU阈值: {match_iou}")
        lines.append(f"面积变化阈值: {area_thresh}")
        lines.append(f"切片尺寸: {tile_size}x{tile_size}, 步长={stride}")
        lines.append("")
        if bounds:
            lines.append(f"重叠区域: {bounds[2]-bounds[0]:.1f}m x {bounds[3]-bounds[1]:.1f}m")
        lines.append(f"影像像素: {arr1.shape[1]}x{arr1.shape[0]}")
        lines.append("")
        lines.append(f"基期检测建筑物: {len(det1)}")
        lines.append(f"检测期检测建筑物: {len(det2)}")
        lines.append(f"匹配(两期均有): {len(matches)}")
        lines.append(f"新增建筑物: {len(new_buildings)}")
        lines.append(f"消失建筑物: {len(disappeared)}")
        lines.append("")

        if new_buildings:
            lines.append("--- 新增建筑物 ---")
            for nb in new_buildings:
                lines.append(f"  {nb['class']} conf={nb['conf']} bbox={nb['bbox']}")

        if disappeared:
            lines.append("--- 消失建筑物 ---")
            for da in disappeared:
                lines.append(f"  {da['class']} conf={da['conf']} bbox={da['bbox']}")

        lines.append("")
        lines.append(f"总耗时: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
        lines.append(f"任务ID: {task_id}")
        return "\n".join(lines)


def _bbox_overlap(bbox, tile_bbox):
    ix1 = max(bbox[0], tile_bbox[0])
    iy1 = max(bbox[1], tile_bbox[1])
    ix2 = min(bbox[2], tile_bbox[2])
    iy2 = min(bbox[3], tile_bbox[3])
    return max(0, ix2 - ix1) > 0 and max(0, iy2 - iy1) > 0


def _shift_poly_to_tile(poly, x0, y0):
    if poly is None or len(poly) < 3:
        return None
    shifted = poly.copy()
    shifted[:, 0] -= x0
    shifted[:, 1] -= y0
    return shifted


def _draw_mask_poly_on_tile(canvas, poly, color, alpha=0.35):
    if poly is None or len(poly) < 3:
        return
    pts = poly.astype(np.int32)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    cv2.polylines(canvas, [pts], True, color, 2)