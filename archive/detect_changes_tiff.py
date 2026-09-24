#!/usr/bin/env python3
"""
大尺寸TIFF影像建筑物变化检测
流程: GDAL配准裁剪 -> 切片YOLO推理 -> 跨切片NMS -> 两期匹配 -> 切片对比可视化.

基期(第一期)为比对基准, 检测期(第二期)新增建筑物标红色
输出: 每个切片的左右对比图(左=基期, 右=检测期), 新增建筑标红, 消失标紫
"""

import os
import time

os.environ["PROJ_LIB"] = "/home/fumu/conda_disk/anaconda3/envs/wrj_torch/share/proj"
os.environ["PROJ_DATA"] = "/home/fumu/conda_disk/anaconda3/envs/wrj_torch/share/proj"
from pathlib import Path

import cv2
import numpy as np
from osgeo import gdal

from ultralytics import YOLO

PERIOD1_PATH = "/home/fumu/conda_disk/baiduwangpan/无人机违建_test/第一期/第一期.tif"
PERIOD2_PATH = "/home/fumu/conda_disk/baiduwangpan/无人机违建_test/第二期/第二期.tif"
MODEL_PATH = "/home/fumu/PycharmProjects/ultralytics-main/runs/segment/train-3/weights/best.pt"
OUTPUT_DIR = "/home/fumu/conda_disk/change_detection_output"

TARGET_RES = 0.2
TILE_SIZE = 1024
STRIDE = 800
CONF_THRESH = 0.5
NMS_IOU = 0.5
MATCH_IOU = 0.3
VIS_TILE_SIZE = 1024


def get_overlap_bounds(path1, path2):
    gdal.UseExceptions()
    ds1 = gdal.Open(path1)
    ds2 = gdal.Open(path2)
    gt1 = ds1.GetGeoTransform()
    gt2 = ds2.GetGeoTransform()

    def image_bounds(ds, gt):
        xmin = gt[0]
        xmax = gt[0] + ds.RasterXSize * gt[1]
        ymax = gt[3]
        ymin = gt[3] + ds.RasterYSize * gt[5]
        return xmin, ymin, xmax, ymax

    b1 = image_bounds(ds1, gt1)
    b2 = image_bounds(ds2, gt2)
    ds1 = None
    ds2 = None

    oxmin = max(b1[0], b2[0])
    oymin = max(b1[1], b2[1])
    oxmax = min(b1[2], b2[2])
    oymax = min(b1[3], b2[3])

    if oxmin >= oxmax or oymin >= oymax:
        raise ValueError("两期影像无重叠区域")

    w_m = oxmax - oxmin
    h_m = oymax - oymin
    print(f"重叠区域: X=[{oxmin:.2f},{oxmax:.2f}] Y=[{oymin:.2f},{oymax:.2f}]")
    print(f"重叠尺寸: {w_m:.1f}m x {h_m:.1f}m")
    print(f"预计像素: {int(w_m / TARGET_RES)} x {int(h_m / TARGET_RES)}")
    return oxmin, oymin, oxmax, oymax


def warp_to_overlap(src_path, bounds, target_res, out_path):
    xmin, ymin, xmax, ymax = bounds
    ds_src = gdal.Open(src_path)
    src_nodata = ds_src.GetRasterBand(1).GetNoDataValue()
    n_bands = ds_src.RasterCount
    ds_src = None

    warp_kwargs = {
        "format": "GTiff",
        "outputBounds": (xmin, ymin, xmax, ymax),
        "xRes": target_res,
        "yRes": target_res,
        "resampleAlg": "bilinear",
        "outputType": gdal.GDT_Byte,
        "creationOptions": ["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES", "PHOTOMETRIC=RGB"],
    }

    if src_nodata is not None:
        warp_kwargs["srcNodata"] = [src_nodata] * n_bands
        warp_kwargs["dstNodata"] = [0] * n_bands
        print(f"  srcNodata={src_nodata} -> dstNodata=0")

    opts = gdal.WarpOptions(**warp_kwargs)
    result = gdal.Warp(out_path, src_path, options=opts)
    if result is None:
        raise RuntimeError(f"GDAL Warp失败: {src_path}")
    result = None
    ds_check = gdal.Open(out_path)
    print(f"  Warp完成: {ds_check.RasterXSize}x{ds_check.RasterYSize} pixels")
    ds_check = None


def read_as_bgr_uint8(tiff_path):
    gdal.UseExceptions()
    ds = gdal.Open(tiff_path)
    nodata_val = ds.GetRasterBand(1).GetNoDataValue()
    bands = []
    for i in range(1, ds.RasterCount + 1):
        bands.append(ds.GetRasterBand(i).ReadAsArray())
    arr = np.stack(bands, axis=2)
    print(f"  原始: shape={arr.shape}, dtype={arr.dtype}, range=[{arr.min()}, {arr.max()}]")
    if nodata_val is not None:
        if 0 <= nodata_val <= 255:
            nodata_mask = np.all(arr == int(nodata_val), axis=2)
        else:
            nodata_mask = np.any(arr == nodata_val, axis=2)
        nd_count = np.sum(nodata_mask)
        valid_pct = (1 - nd_count / (arr.shape[0] * arr.shape[1])) * 100
        print(f"  NoData={nodata_val}, 有效像素占比={valid_pct:.1f}%")
        arr[nodata_mask] = 0

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    arr = arr[:, :, ::-1].copy()
    print(f"  BGR uint8: shape={arr.shape}, dtype={arr.dtype}, range=[{arr.min()}, {arr.max()}]")

    gt = ds.GetGeoTransform()
    crs = ds.GetProjection()
    ds = None
    return arr, gt, crs


def crop_black_border(arr1, arr2, gt1, threshold=2):
    valid1 = np.any(arr1 > threshold, axis=2)
    valid2 = np.any(arr2 > threshold, axis=2)
    valid = valid1 & valid2

    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)
    if not rows.any() or not cols.any():
        print("  警告: 无有效像素区域")
        return arr1, arr2, gt1

    rmin = np.where(rows)[0][0]
    rmax = np.where(rows)[0][-1] + 1
    cmin = np.where(cols)[0][0]
    cmax = np.where(cols)[0][-1] + 1

    original_pct = valid.sum() / (arr1.shape[0] * arr1.shape[1]) * 100
    print(f"  黑边裁剪: 原始 {arr1.shape[1]}x{arr1.shape[0]} -> 有效区域 [{rmin}:{rmax}, {cmin}:{cmax}]")
    print(f"  裁剪后: {cmax - cmin}x{rmax - rmin}, 有效像素占比: {original_pct:.1f}% -> 100%")

    arr1 = arr1[rmin:rmax, cmin:cmax]
    arr2 = arr2[rmin:rmax, cmin:cmax]

    gt_new = list(gt1)
    gt_new[0] = gt_new[0] + cmin * gt_new[1]
    gt_new[3] = gt_new[3] + rmin * gt_new[5]
    gt_new = tuple(gt_new)

    return arr1, arr2, gt_new


def tile_positions(h, w, tile_size, stride):
    ys = list(range(0, max(1, h - tile_size + 1), stride))
    xs = list(range(0, max(1, w - tile_size + 1), stride))
    if not ys or ys[-1] + tile_size < h:
        ys.append(max(0, h - tile_size))
    if not xs or xs[-1] + tile_size < w:
        xs.append(max(0, w - tile_size))
    ys = sorted(set(ys))
    xs = sorted(set(xs))
    return [(y, x) for y in ys for x in xs]


def detect_tiled(model, img, tile_size, stride, conf_thresh):
    h, w = img.shape[:2]
    positions = tile_positions(h, w, tile_size, stride)
    print(f"  图片 {w}x{h}, 切片数 {len(positions)}")

    all_dets = []
    t0 = time.time()
    for idx, (y0, x0) in enumerate(positions):
        tile = img[y0 : y0 + tile_size, x0 : x0 + tile_size]
        th, tw = tile.shape[:2]
        pad_h = tile_size - th
        pad_w = tile_size - tw
        if pad_h > 0 or pad_w > 0:
            tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

        results = model(tile, conf=conf_thresh, verbose=False)

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for i, box in enumerate(r.boxes):
                cls_name = model.names[int(box.cls[0])]
                xyxy = box.xyxy[0].cpu().numpy()
                conf_val = float(box.conf[0])

                x1 = xyxy[0] + x0
                y1 = xyxy[1] + y0
                x2 = xyxy[2] + x0
                y2 = xyxy[3] + y0

                if x2 > w + 5 or y2 > h + 5 or x1 < -5 or y1 < -5:
                    continue
                x1 = max(0, min(w, x1))
                y1 = max(0, min(h, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))

                bbox = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]

                mask_poly = None
                mask_np = None
                if r.masks is not None and i < len(r.masks):
                    mask_np = r.masks.data[i].cpu().numpy()
                    try:
                        if hasattr(r.masks, "xy") and len(r.masks.xy) > i:
                            poly = r.masks.xy[i].copy()
                            poly[:, 0] += x0
                            poly[:, 1] += y0
                            if pad_h > 0 or pad_w > 0:
                                poly[:, 0] = np.clip(poly[:, 0], 0, w)
                                poly[:, 1] = np.clip(poly[:, 1], 0, h)
                            mask_poly = poly
                    except Exception:
                        mask_poly = None

                all_dets.append(
                    {
                        "class": cls_name,
                        "bbox": bbox,
                        "conf": round(conf_val, 3),
                        "mask": mask_np,
                        "mask_poly": mask_poly,
                    }
                )

        if (idx + 1) % 50 == 0 or idx == len(positions) - 1:
            elapsed = time.time() - t0
            print(f"    {idx + 1}/{len(positions)}, {elapsed:.1f}s")

    print(f"  原始检测数: {len(all_dets)}")
    return all_dets


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


def nms(detections, iou_thresh):
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
    print(f"  NMS: {len(dets)} -> {len(keep)}")
    return keep


def find_changes(det_a, det_b, iou_thresh=MATCH_IOU):
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


def bbox_overlap(bbox, tile_bbox):
    ix1 = max(bbox[0], tile_bbox[0])
    iy1 = max(bbox[1], tile_bbox[1])
    ix2 = min(bbox[2], tile_bbox[2])
    iy2 = min(bbox[3], tile_bbox[3])
    return max(0, ix2 - ix1) > 0 and max(0, iy2 - iy1) > 0


def shift_poly_to_tile(poly, x0, y0):
    if poly is None or len(poly) < 3:
        return None
    shifted = poly.copy()
    shifted[:, 0] -= x0
    shifted[:, 1] -= y0
    return shifted


def draw_mask_poly_on_tile(canvas, poly, color, alpha=0.35):
    if poly is None or len(poly) < 3:
        return
    pts = poly.astype(np.int32)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    cv2.polylines(canvas, [pts], True, color, 2)


def create_comparison_tiles(arr1, arr2, det1, matches, new_buildings, disappeared, output_dir, vis_tile_size):
    h, w = arr1.shape[:2]
    tile_dir = os.path.join(output_dir, "comparison_tiles")
    Path(tile_dir).mkdir(parents=True, exist_ok=True)

    {id(m["box_b"]) for m in matches}
    disappeared_ids = {id(d) for d in disappeared}
    {id(nb) for nb in new_buildings}

    n_rows = h // vis_tile_size
    n_cols = w // vis_tile_size
    positions = []
    row = 0
    for y in range(0, n_rows * vis_tile_size, vis_tile_size):
        col = 0
        for x in range(0, n_cols * vis_tile_size, vis_tile_size):
            positions.append((y, x, row, col))
            col += 1
        row += 1

    if n_rows * vis_tile_size < h or n_cols * vis_tile_size < w:
        margin_h = h - n_rows * vis_tile_size
        margin_w = w - n_cols * vis_tile_size
        print(f"  跳过边缘碎片: 底部{margin_h}px, 右侧{margin_w}px (不足一个完整切片)")

    total = len(positions)
    print(f"  生成 {total} 个对比切片 ({vis_tile_size}x{vis_tile_size}), 网格 {n_rows}行 x {n_cols}列")

    has_change_tiles = 0
    for idx, (y0, x0, row, col) in enumerate(positions):
        crop1 = arr1[y0 : y0 + vis_tile_size, x0 : x0 + vis_tile_size]
        crop2 = arr2[y0 : y0 + vis_tile_size, x0 : x0 + vis_tile_size]

        tile_bbox = [x0, y0, x0 + vis_tile_size, y0 + vis_tile_size]

        canvas_a = crop1.copy()
        canvas_b = crop2.copy()

        tile_has_change = False

        for da in det1:
            if not bbox_overlap(da["bbox"], tile_bbox):
                continue
            is_disappeared = id(da) in disappeared_ids
            color = (128, 0, 128) if is_disappeared else (0, 200, 0)
            label = "GONE" if is_disappeared else ""
            poly = shift_poly_to_tile(da.get("mask_poly"), x0, y0)
            draw_mask_poly_on_tile(canvas_a, poly, color, alpha=0.3 if is_disappeared else 0.2)
            x1t = max(0, int(da["bbox"][0]) - x0)
            y1t = max(0, int(da["bbox"][1]) - y0)
            x2t = min(vis_tile_size, int(da["bbox"][2]) - x0)
            y2t = min(vis_tile_size, int(da["bbox"][3]) - y0)
            cv2.rectangle(canvas_a, (x1t, y1t), (x2t, y2t), color, 2)
            if label:
                cv2.putText(canvas_a, label, (x1t, max(12, y1t - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            if is_disappeared:
                tile_has_change = True

        for nb in new_buildings:
            if not bbox_overlap(nb["bbox"], tile_bbox):
                continue
            color = (0, 0, 255)
            poly = shift_poly_to_tile(nb.get("mask_poly"), x0, y0)
            draw_mask_poly_on_tile(canvas_b, poly, color, alpha=0.5)
            x1t = max(0, int(nb["bbox"][0]) - x0)
            y1t = max(0, int(nb["bbox"][1]) - y0)
            x2t = min(vis_tile_size, int(nb["bbox"][2]) - x0)
            y2t = min(vis_tile_size, int(nb["bbox"][3]) - y0)
            cv2.rectangle(canvas_b, (x1t, y1t), (x2t, y2t), color, 3)
            cv2.putText(canvas_b, "NEW!", (x1t, max(12, y1t - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            tile_has_change = True

        for m in matches:
            db = m["box_b"]
            if not bbox_overlap(db["bbox"], tile_bbox):
                continue
            color = (0, 200, 0)
            poly = shift_poly_to_tile(db.get("mask_poly"), x0, y0)
            draw_mask_poly_on_tile(canvas_b, poly, color, alpha=0.2)
            x1t = max(0, int(db["bbox"][0]) - x0)
            y1t = max(0, int(db["bbox"][1]) - y0)
            x2t = min(vis_tile_size, int(db["bbox"][2]) - x0)
            y2t = min(vis_tile_size, int(db["bbox"][3]) - y0)
            cv2.rectangle(canvas_b, (x1t, y1t), (x2t, y2t), color, 2)

        comparison = cv2.hconcat([canvas_a, canvas_b])
        cv2.line(comparison, (vis_tile_size, 0), (vis_tile_size, vis_tile_size), (255, 255, 255), 2)
        cv2.putText(comparison, "基期(第一期)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(
            comparison, "检测期(第二期)", (vis_tile_size + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.putText(
            comparison,
            f"Row{row} Col{col}",
            (10, vis_tile_size - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )

        change_tag = "_change" if tile_has_change else ""
        tile_path = os.path.join(tile_dir, f"tile_r{row}_c{col}{change_tag}.png")
        cv2.imwrite(tile_path, comparison)

        if tile_has_change:
            has_change_tiles += 1

        if (idx + 1) % 20 == 0 or idx == total - 1:
            print(f"    {idx + 1}/{total} 切片完成, 有变化={has_change_tiles}")

    return tile_dir, has_change_tiles, total


def create_overview(arr1, arr2, det1, matches, new_buildings, disappeared, output_dir, scale=0.1):
    h, w = arr1.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)

    thumb1 = cv2.resize(arr1, (new_w, new_h), interpolation=cv2.INTER_AREA)
    thumb2 = cv2.resize(arr2, (new_w, new_h), interpolation=cv2.INTER_AREA)

    for nb in new_buildings:
        x1, y1, x2, y2 = [int(v * scale) for v in nb["bbox"]]
        cv2.rectangle(thumb2, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(thumb2, "NEW", (x1, max(8, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    for da in disappeared:
        x1, y1, x2, y2 = [int(v * scale) for v in da["bbox"]]
        cv2.rectangle(thumb1, (x1, y1), (x2, y2), (128, 0, 128), 2)

    for m in matches:
        db = m["box_b"]
        x1, y1, x2, y2 = [int(v * scale) for v in db["bbox"]]
        cv2.rectangle(thumb2, (x1, y1), (x2, y2), (0, 200, 0), 1)

    overview = cv2.hconcat([thumb1, thumb2])
    cv2.line(overview, (new_w, 0), (new_w, new_h), (255, 255, 255), 3)
    cv2.putText(overview, "基期(第一期)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(overview, "检测期(第二期)", (new_w + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(
        overview,
        "Red=NEW! Purple=GONE Green=Existing",
        (10, new_h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )

    overview_path = os.path.join(output_dir, "overview_comparison.png")
    cv2.imwrite(overview_path, overview)
    print(f"  总览保存: {overview_path} ({overview.shape[1]}x{overview.shape[0]})")
    return overview_path


def write_geotiff(out_path, data, gt, crs, dtype=gdal.GDT_Byte):
    driver = gdal.GetDriverByName("GTiff")
    if data.ndim == 2:
        h, w = data.shape
        bands = 1
    else:
        h, w = data.shape[:2]
        bands = data.shape[2]

    ds = driver.Create(
        out_path, w, h, bands, dtype, options=["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES", "PHOTOMETRIC=RGB"]
    )
    ds.SetGeoTransform(gt)
    try:
        ds.SetProjection(crs)
    except Exception as e:
        print(f"  警告: 无法写入CRS投影信息({e}), 将保存为无投影GeoTIFF")

    if bands == 1:
        ds.GetRasterBand(1).WriteArray(data)
    else:
        data_rgb = data[:, :, ::-1].copy()
        for i in range(bands):
            ds.GetRasterBand(i + 1).WriteArray(data_rgb[:, :, i])

    try:
        ds.FlushCache()
    except Exception as e:
        print(f"  警告: FlushCache时CRS写入异常({e}), 数据已写入")
    ds = None
    print(f"  保存GeoTIFF: {out_path} ({w}x{h}, {bands}bands)")


def main():
    t_start = time.time()
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("大尺寸TIFF建筑物变化检测")
    print("=" * 60)
    print(f"基期: {PERIOD1_PATH}")
    print(f"检测期: {PERIOD2_PATH}")
    print(f"模型: {MODEL_PATH}")
    print(f"输出: {OUTPUT_DIR}")
    print(f"分辨率: {TARGET_RES}m, 推理切片: {TILE_SIZE}x{TILE_SIZE}, 步长={STRIDE}")
    print(f"对比切片: {VIS_TILE_SIZE}x{VIS_TILE_SIZE}")

    print("\n[1] 计算重叠区域...")
    bounds = get_overlap_bounds(PERIOD1_PATH, PERIOD2_PATH)

    print("\n[2] GDAL配准+裁剪...")
    p1_warped = os.path.join(OUTPUT_DIR, "period1_aligned.tif")
    p2_warped = os.path.join(OUTPUT_DIR, "period2_aligned.tif")
    warp_to_overlap(PERIOD1_PATH, bounds, TARGET_RES, p1_warped)
    warp_to_overlap(PERIOD2_PATH, bounds, TARGET_RES, p2_warped)

    print("\n[3] 读取为BGR uint8...")
    arr1, gt_out, crs_out = read_as_bgr_uint8(p1_warped)
    arr2, _gt2, _crs2 = read_as_bgr_uint8(p2_warped)
    h = min(arr1.shape[0], arr2.shape[0])
    w = min(arr1.shape[1], arr2.shape[1])
    arr1 = arr1[:h, :w]
    arr2 = arr2[:h, :w]

    print("\n[3.5] 去除黑边...")
    arr1, arr2, gt_out = crop_black_border(arr1, arr2, gt_out)

    print("\n[4] 加载YOLO模型...")
    model = YOLO(MODEL_PATH)

    print("\n[5] 第一期(基期)切片推理...")
    raw_dets1 = detect_tiled(model, arr1, TILE_SIZE, STRIDE, CONF_THRESH)
    det1 = nms(raw_dets1, NMS_IOU)

    print("\n[6] 第二期(检测期)切片推理...")
    raw_dets2 = detect_tiled(model, arr2, TILE_SIZE, STRIDE, CONF_THRESH)
    det2 = nms(raw_dets2, NMS_IOU)

    print("\n[7] 变化检测匹配...")
    matches, new_buildings, disappeared = find_changes(det1, det2, MATCH_IOU)

    print(f"\n  基期建筑物: {len(det1)}")
    print(f"  检测期建筑物: {len(det2)}")
    print(f"  匹配(两期均有): {len(matches)}")
    print(f"  新增建筑物: {len(new_buildings)}")
    print(f"  消失建筑物: {len(disappeared)}")

    print("\n[8] 生成切片对比图...")
    _tile_dir, has_change, total_tiles = create_comparison_tiles(
        arr1, arr2, det1, matches, new_buildings, disappeared, OUTPUT_DIR, VIS_TILE_SIZE
    )
    print(f"  有变化的切片: {has_change}/{total_tiles}")

    print("\n[9] 生成总览图...")
    create_overview(arr1, arr2, det1, matches, new_buildings, disappeared, OUTPUT_DIR, scale=0.1)

    print("\n[10] 保存建筑物掩膜GeoTIFF...")
    new_mask = np.zeros(arr2.shape[:2], dtype=np.uint8)
    for nb in new_buildings:
        if nb.get("mask_poly") is not None and len(nb["mask_poly"]) >= 3:
            cv2.fillPoly(new_mask, [nb["mask_poly"].astype(np.int32)], 255)
        else:
            x1, y1, x2, y2 = [int(v) for v in nb["bbox"]]
            new_mask[max(0, y1) : y2, max(0, x1) : x2] = 255
    write_geotiff(os.path.join(OUTPUT_DIR, "new_buildings_mask.tif"), new_mask, gt_out, crs_out)

    print("\n[11] 生成报告...")
    report = []
    report.append("=" * 60)
    report.append("建筑物变化检测报告")
    report.append("=" * 60)
    report.append(f"基期(第一期): {PERIOD1_PATH}")
    report.append(f"检测期(第二期): {PERIOD2_PATH}")
    report.append(f"模型: {MODEL_PATH}")
    report.append(f"目标分辨率: {TARGET_RES}m/像素")
    report.append(f"推理切片: {TILE_SIZE}x{TILE_SIZE}, 步长={STRIDE}")
    report.append(f"对比切片: {VIS_TILE_SIZE}x{VIS_TILE_SIZE}")
    report.append(f"置信度阈值: {CONF_THRESH}")
    report.append(f"NMS IoU阈值: {NMS_IOU}")
    report.append(f"匹配IoU阈值: {MATCH_IOU}")
    report.append("")
    report.append(f"重叠区域: {bounds[2] - bounds[0]:.1f}m x {bounds[3] - bounds[1]:.1f}m")
    report.append(f"影像像素: {arr1.shape[1]}x{arr1.shape[0]}")
    report.append("")
    report.append(f"基期检测建筑物: {len(det1)}")
    report.append(f"检测期检测建筑物: {len(det2)}")
    report.append(f"匹配(两期均有): {len(matches)}")
    report.append(f"新增建筑物: {len(new_buildings)}")
    report.append(f"消失建筑物: {len(disappeared)}")
    report.append(f"有变化的切片: {has_change}/{total_tiles}")
    report.append("")

    if new_buildings:
        report.append("--- 新增建筑物 ---")
        for nb in new_buildings:
            report.append(f"  {nb['class']} conf={nb['conf']} bbox={nb['bbox']}")

    if disappeared:
        report.append("--- 消失建筑物 ---")
        for da in disappeared:
            report.append(f"  {da['class']} conf={da['conf']} bbox={da['bbox']}")

    report.append("")
    report.append(f"总耗时: {time.time() - t_start:.1f}s")
    report.append(f"输出目录: {OUTPUT_DIR}")
    report.append("")
    report.append("输出文件:")
    report.append("  overview_comparison.png     - 全图总览(缩小10倍,左基期右检测期)")
    report.append("  comparison_tiles/           - 切片对比图目录(每个1024x2048)")
    report.append("  comparison_tiles/ *_change.png - 有变化的切片(重点查看)")
    report.append("  new_buildings_mask.tif      - 新增建筑物掩膜GeoTIFF")
    report.append("  period1_aligned.tif         - 基期配准裁剪影像")
    report.append("  period2_aligned.tif         - 检测期配准裁剪影像")

    report_path = os.path.join(OUTPUT_DIR, "detect_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"\n报告保存: {report_path}")
    elapsed = time.time() - t_start
    print(f"总耗时: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
