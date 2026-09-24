import logging
import os

import cv2
import numpy as np

try:
    from osgeo import gdal, ogr, osr

    gdal.UseExceptions()
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False

try:
    import tifffile

    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

logger = logging.getLogger(__name__)


def get_overlap_bounds(path1, path2):
    if not HAS_GDAL:
        raise RuntimeError("GDAL is required for TIFF overlap calculation")
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

    logger.info(f"重叠区域: X=[{oxmin:.2f},{oxmax:.2f}] Y=[{oymin:.2f},{oymax:.2f}]")
    return oxmin, oymin, oxmax, oymax


def warp_to_overlap(src_path, bounds, target_res, out_path):
    if not HAS_GDAL:
        raise RuntimeError("GDAL is required for warp")
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

    opts = gdal.WarpOptions(**warp_kwargs)
    result = gdal.Warp(out_path, src_path, options=opts)
    if result is None:
        raise RuntimeError(f"GDAL Warp失败: {src_path}")
    result = None


def read_as_bgr_uint8(tiff_path):
    if HAS_GDAL:
        ds = gdal.Open(tiff_path)
        nodata_val = ds.GetRasterBand(1).GetNoDataValue()
        bands = []
        for i in range(1, ds.RasterCount + 1):
            bands.append(ds.GetRasterBand(i).ReadAsArray())
        arr = np.stack(bands, axis=2)
        logger.info(f"原始: shape={arr.shape}, dtype={arr.dtype}")

        if nodata_val is not None:
            if 0 <= nodata_val <= 255:
                nodata_mask = np.all(arr == int(nodata_val), axis=2)
            else:
                nodata_mask = np.any(arr == nodata_val, axis=2)
            nd_count = np.sum(nodata_mask)
            valid_pct = (1 - nd_count / (arr.shape[0] * arr.shape[1])) * 100
            logger.info(f"NoData={nodata_val}, 有效像素占比={valid_pct:.1f}%")
            arr[nodata_mask] = 0

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        arr = arr[:, :, ::-1].copy()

        gt = ds.GetGeoTransform()
        crs = ds.GetProjection()
        ds = None
        return arr, gt, crs

    img = cv2.imread(tiff_path, cv2.IMREAD_COLOR)
    if img is not None:
        return img, None, None

    if HAS_TIFFFILE:
        arr = tifffile.imread(tiff_path)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        elif arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr, None, None

    raise RuntimeError(f"无法读取影像: {tiff_path}")


def crop_black_border(arr1, arr2, gt1, threshold=2):
    valid1 = np.any(arr1 > threshold, axis=2)
    valid2 = np.any(arr2 > threshold, axis=2)
    valid = valid1 & valid2

    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)
    if not rows.any() or not cols.any():
        logger.warning("无有效像素区域")
        return arr1, arr2, gt1

    rmin = np.where(rows)[0][0]
    rmax = np.where(rows)[0][-1] + 1
    cmin = np.where(cols)[0][0]
    cmax = np.where(cols)[0][-1] + 1

    arr1 = arr1[rmin:rmax, cmin:cmax]
    arr2 = arr2[rmin:rmax, cmin:cmax]

    if gt1 is not None:
        gt_new = list(gt1)
        gt_new[0] = gt_new[0] + cmin * gt_new[1]
        gt_new[3] = gt_new[3] + rmin * gt_new[5]
        gt_new = tuple(gt_new)
        return arr1, arr2, gt_new

    return arr1, arr2, gt1


def write_geotiff(out_path, data, gt, crs, dtype=None):
    if not HAS_GDAL:
        logger.warning("GDAL不可用，无法写入GeoTIFF")
        return
    if dtype is None:
        dtype = gdal.GDT_Byte

    driver = gdal.GetDriverByName("GTiff")
    if data.ndim == 2:
        h, w = data.shape
        bands = 1
    else:
        h, w = data.shape[:2]
        bands = data.shape[2]

    creation_opts = ["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES"]
    if bands >= 3:
        creation_opts.append("PHOTOMETRIC=RGB")
    ds = driver.Create(out_path, w, h, bands, dtype, options=creation_opts)
    if gt is not None:
        ds.SetGeoTransform(gt)
    if crs is not None:
        try:
            ds.SetProjection(crs)
        except Exception as e:
            logger.warning(f"无法写入CRS投影信息({e})")

    if bands == 1:
        ds.GetRasterBand(1).WriteArray(data)
    else:
        data_rgb = data[:, :, ::-1].copy()
        for i in range(bands):
            ds.GetRasterBand(i + 1).WriteArray(data_rgb[:, :, i])

    try:
        ds.FlushCache()
    except Exception:
        pass
    ds = None
    logger.info(f"保存GeoTIFF: {out_path} ({w}x{h}, {bands}bands)")


def write_changes_to_shapefile(changes, gt, crs, output_dir, task_id):
    if not HAS_GDAL:
        logger.warning("GDAL不可用，无法生成Shapefile，仅输出JSON")
        return None

    shp_dir = os.path.join(output_dir, f"instance_changes_{task_id}")
    os.makedirs(shp_dir, exist_ok=True)
    shp_path = os.path.join(shp_dir, f"instance_changes_{task_id}.shp")

    drv = ogr.GetDriverByName("ESRI Shapefile")
    ds = drv.CreateDataSource(shp_path)
    if ds is None:
        logger.error(f"无法创建Shapefile: {shp_path}")
        return None

    srs = osr.SpatialReference()
    if crs:
        srs.ImportFromWkt(crs)
    layer = ds.CreateLayer("changes", srs, ogr.wkbPolygon)

    fld_id = ogr.FieldDefn("id", ogr.OFTInteger)
    fld_pre = ogr.FieldDefn("pre_class", ogr.OFTInteger)
    fld_post = ogr.FieldDefn("post_class", ogr.OFTInteger)
    layer.CreateField(fld_id)
    layer.CreateField(fld_pre)
    layer.CreateField(fld_post)

    if gt is not None:
        pixel_to_geo = _make_pixel_to_geo(gt)
    else:
        pixel_to_geo = None

    type_to_pre_post = {
        "new": (0, 1),
        "removed": (1, 0),
        "expanded": (1, 1),
    }

    for idx, ch in enumerate(changes.get("changes", []), start=1):
        bbox = ch["bbox"]
        change_type = ch["type"]
        pre_class, post_class = type_to_pre_post.get(change_type, (0, 1))

        ring = ogr.Geometry(ogr.wkbLinearRing)
        corners = [
            (bbox[0], bbox[1]),
            (bbox[2], bbox[1]),
            (bbox[2], bbox[3]),
            (bbox[0], bbox[3]),
            (bbox[0], bbox[1]),
        ]
        for cx, cy in corners:
            if pixel_to_geo:
                gx, gy = pixel_to_geo(cx, cy)
                ring.AddPoint(gx, gy)
            else:
                ring.AddPoint(cx, cy)

        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetGeometry(poly)
        feat.SetField("id", idx)
        feat.SetField("pre_class", pre_class)
        feat.SetField("post_class", post_class)
        layer.CreateFeature(feat)
        feat = None

    ds = None
    logger.info(f"Shapefile已创建: {shp_path}")
    return shp_dir


def _make_pixel_to_geo(gt):
    def transform(px, py):
        gx = gt[0] + px * gt[1] + py * gt[2]
        gy = gt[3] + px * gt[4] + py * gt[5]
        return gx, gy

    return transform
