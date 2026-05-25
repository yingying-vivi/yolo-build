import os
import logging
from concurrent.futures import ThreadPoolExecutor
from flask import Blueprint, request, jsonify, send_file

from ..services.change_service import ChangeDetectionService
from ..utils.task_manager import TaskManager
from ..models.model_config_loader import ModelConfigLoader
from ..config import Config

logger = logging.getLogger(__name__)

image_bp = Blueprint('image', __name__, url_prefix='/api/images')

RESULT_ROOT = Config.API_RESULTS_DIR
_executor = ThreadPoolExecutor(max_workers=Config.THREAD_POOL_MAX_WORKERS)
_cds = ChangeDetectionService()


def _run_change_task(task_id, t1_path, t2_path,
                     confidence_threshold, min_area_pixels,
                     iou_threshold, area_change_threshold, crop_size,
                     model_path, use_adaptive_threshold):
    try:
        TaskManager.start_task(task_id, message='正在进行变化检测')

        tile_size = crop_size
        stride = int(tile_size * 0.8)
        nms_iou_threshold = Config.DEFAULT_NMS_IOU_THRESHOLD
        vis_tile_size = Config.DEFAULT_VIS_TILE_SIZE
        target_res = Config.DEFAULT_TARGET_RES

        if model_path is None:
            config_loader = ModelConfigLoader()
            model_path = config_loader.get_yolo_seg_model_path()

        result = _cds.run_change_detection(
            t1_path=t1_path,
            t2_path=t2_path,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            match_iou_threshold=iou_threshold,
            area_change_threshold=area_change_threshold,
            tile_size=tile_size,
            stride=stride,
            target_res=target_res,
            vis_tile_size=vis_tile_size,
            output_dir=RESULT_ROOT,
            task_id=task_id,
        )
        TaskManager.complete_task(
            task_id,
            output=result.get('zip_path'),
            result_json=result.get('changes_dict'),
            message='变化检测完成',
        )
    except Exception as e:
        logger.exception('变化检测任务失败')
        TaskManager.fail_task(task_id, error=str(e), message='变化检测失败')


@image_bp.route('/segment', methods=['POST'])
def image_segmentation():
    data = request.get_json()
    if not data:
        return jsonify({'error': '未提供有效的JSON数据'}), 415

    image_path = data.get('image_path')
    model_path = data.get('model_path')
    if not image_path or not model_path:
        return jsonify({'error': '必须提供image_path和model_path参数'}), 400
    if not os.path.exists(image_path):
        return jsonify({'error': f'图像文件不存在: {image_path}'}), 404
    if not os.path.exists(model_path):
        return jsonify({'error': f'YOLO-seg 模型文件不存在: {model_path}'}), 404

    confidence_threshold = float(data.get('confidence_threshold', 0.3))
    crop_size = data.get('crop_size')
    if crop_size in (None, ''):
        crop_size = 4096
    else:
        try:
            crop_size = int(float(crop_size))
            if crop_size < 256:
                crop_size = 256
        except Exception:
            crop_size = 4096

    min_area_pixels = data.get('min_area_pixels')
    if min_area_pixels in (None, ''):
        min_area_pixels = 500
    else:
        try:
            min_area_pixels = int(float(min_area_pixels))
            if min_area_pixels < 0:
                min_area_pixels = 0
        except Exception:
            min_area_pixels = 500

    use_adaptive_threshold = data.get('use_adaptive_threshold')
    if use_adaptive_threshold in (None, ''):
        use_adaptive_threshold = False
    else:
        use_adaptive_threshold = str(use_adaptive_threshold).lower() in ('true', '1', 'yes')

    task_id = TaskManager.create_task()
    return jsonify({
        'status': 1,
        'taskId': task_id,
        'message': '分割接口暂未实现，请使用变化检测接口',
    }), 200


@image_bp.route('/segment/status/<task_id>', methods=['GET'])
def get_segmentation_status(task_id):
    task_info = TaskManager.get_task(task_id)
    if not task_info:
        return jsonify({'error': '任务不存在'}), 404
    return jsonify(task_info)


@image_bp.route('/change', methods=['POST'])
def change_detection():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '未提供有效的JSON数据'}), 415

        t1_path = data.get('t1_path')
        t2_path = data.get('t2_path')
        if not t1_path or not t2_path:
            return jsonify({'error': '需要 t1_path 和 t2_path 参数'}), 400

        if not os.path.exists(t1_path):
            return jsonify({'error': f'T1 文件不存在: {t1_path}'}), 404
        if not os.path.exists(t2_path):
            return jsonify({'error': f'T2 文件不存在: {t2_path}'}), 404

        confidence_threshold = data.get('confidence_threshold')
        if confidence_threshold in (None, ''):
            confidence_threshold = 0.3
        else:
            try:
                confidence_threshold = float(confidence_threshold)
                if not (0.0 < confidence_threshold < 1.0):
                    confidence_threshold = 0.3
            except Exception:
                confidence_threshold = 0.3

        min_area_pixels = data.get('min_area_pixels')
        if min_area_pixels in (None, ''):
            min_area_pixels = 500
        else:
            try:
                min_area_pixels = int(float(min_area_pixels))
                if min_area_pixels < 1:
                    min_area_pixels = 1
            except Exception:
                min_area_pixels = 500

        iou_threshold = data.get('iou_threshold')
        if iou_threshold in (None, ''):
            iou_threshold = 0.5
        else:
            try:
                iou_threshold = float(iou_threshold)
                if not (0.0 < iou_threshold < 1.0):
                    iou_threshold = 0.5
            except Exception:
                iou_threshold = 0.5

        area_change_threshold = data.get('area_change_threshold')
        if area_change_threshold in (None, ''):
            area_change_threshold = 0.2
        else:
            try:
                area_change_threshold = float(area_change_threshold)
                if area_change_threshold < 0.0:
                    area_change_threshold = 0.2
            except Exception:
                area_change_threshold = 0.2

        crop_size = data.get('crop_size')
        if crop_size in (None, ''):
            crop_size = 4096
        else:
            try:
                crop_size = int(float(crop_size))
                if crop_size < 256:
                    crop_size = 256
            except Exception:
                crop_size = 4096

        use_adaptive_threshold = data.get('use_adaptive_threshold')
        if use_adaptive_threshold in (None, ''):
            use_adaptive_threshold = False
        else:
            use_adaptive_threshold = str(use_adaptive_threshold).lower() in ('true', '1', 'yes')

        model_path = data.get('model_path')
        if model_path and not os.path.exists(model_path):
            return jsonify({'error': f'模型文件不存在: {model_path}'}), 404

        task_id = TaskManager.create_task()
        _executor.submit(
            _run_change_task,
            task_id, t1_path, t2_path,
            confidence_threshold, min_area_pixels,
            iou_threshold, area_change_threshold, crop_size,
            model_path, use_adaptive_threshold,
        )

        return jsonify({
            'status': 1,
            'taskId': task_id,
            'message': '任务已提交（固定输出 bbox shp），请使用 taskId 查询处理状态',
        }), 200
    except Exception as e:
        return jsonify({'error': f'处理失败: {str(e)}'}), 500


@image_bp.route('/change/status/<task_id>', methods=['GET'])
def get_change_status(task_id):
    task_info = TaskManager.get_task(task_id)
    if not task_info:
        return jsonify({'error': '任务不存在'}), 404
    return jsonify(task_info)


@image_bp.route('/change/result/<task_id>', methods=['GET'])
def get_change_result(task_id):
    task = TaskManager.get_task(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404

    if task.get('status') != 2:
        return jsonify({'error': f'任务未完成或已失败，状态码: {task.get("status")}'}), 400

    result_file = task.get('result_file')
    if not result_file or not os.path.exists(result_file):
        return jsonify({'error': '结果文件不存在'}), 404

    try:
        return send_file(
            result_file,
            as_attachment=True,
            download_name=os.path.basename(result_file),
            mimetype='application/zip',
        )
    except Exception as e:
        return jsonify({'error': f'读取结果文件失败: {str(e)}'}), 500


@image_bp.route('/download', methods=['GET'])
def download_file_by_path():
    try:
        file_path = request.args.get('file_path')
        if not file_path:
            return jsonify({'error': '缺少 file_path 参数'}), 400
        if not os.path.exists(file_path):
            return jsonify({'error': '文件不存在'}), 404

        return send_file(
            file_path,
            as_attachment=True,
            download_name=os.path.basename(file_path),
            mimetype='application/octet-stream',
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500