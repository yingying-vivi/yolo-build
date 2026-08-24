import logging
import os
import shutil
import zipfile

logger = logging.getLogger(__name__)


def zip_shapefile(shp_dir, output_path):
    if shp_dir is None:
        return None
    os.path.basename(shp_dir)
    zip_path = output_path if output_path else shp_dir + ".zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(shp_dir):
            fpath = os.path.join(shp_dir, fname)
            zf.write(fpath, fname)

    logger.info(f"Shapefile ZIP已创建: {zip_path}")
    return zip_path


def safe_delete_path(path):
    try:
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    except Exception as e:
        logger.warning(f"删除失败: {path}, 错误: {e}")


def create_zip_from_files(file_paths, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in file_paths:
            if os.path.exists(fpath):
                zf.write(fpath, os.path.basename(fpath))
    return zip_path
