import json
import os
import threading
import uuid
from typing import Any, Dict, Optional

from ..config import Config

_TASKS_FILE = os.path.join(Config.API_RESULTS_DIR, 'tasks_history.json')


def _load_tasks_from_disk():
    tasks = {}

    if os.path.exists(_TASKS_FILE):
        try:
            with open(_TASKS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                tasks.update(data)
        except Exception:
            pass

    if os.path.isdir(Config.API_RESULTS_DIR):
        for name in os.listdir(Config.API_RESULTS_DIR):
            task_dir = os.path.join(Config.API_RESULTS_DIR, name)
            if not os.path.isdir(task_dir) or name in tasks or name == 'logs':
                continue
            json_path = os.path.join(task_dir, f'instance_changes_{name}.json')
            zip_path = os.path.join(task_dir, f'instance_changes_{name}.zip')
            report_path = os.path.join(task_dir, 'detect_report.txt')

            has_zip = os.path.exists(zip_path)
            has_json = os.path.exists(json_path)
            has_report = os.path.exists(report_path)

            if has_zip or has_json or has_report:
                result_json = None
                if has_json:
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            result_json = json.load(f)
                    except Exception:
                        pass

                if has_zip:
                    tasks[name] = {
                        'taskId': name,
                        'status': 2,
                        'output': None,
                        'error': None,
                        'result_json': result_json,
                        'result_file': zip_path,
                        'message': '变化检测完成',
                    }
                else:
                    tasks[name] = {
                        'taskId': name,
                        'status': 3,
                        'output': None,
                        'error': '任务失败，结果文件未生成',
                        'result_json': result_json,
                        'result_file': None,
                        'message': '变化检测失败',
                    }

    return tasks


def _save_tasks_to_disk(tasks):
    try:
        os.makedirs(os.path.dirname(_TASKS_FILE), exist_ok=True)
        with open(_TASKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


class TaskManager:
    _lock = threading.Lock()
    _tasks: Dict[str, Dict[str, Any]] = _load_tasks_from_disk()

    @classmethod
    def create_task(cls) -> str:
        task_id = uuid.uuid4().hex
        with cls._lock:
            cls._tasks[task_id] = {
                'taskId': task_id,
                'status': 0,
                'output': None,
                'error': None,
                'result_json': None,
            }
            _save_tasks_to_disk(cls._tasks)
        return task_id

    @classmethod
    def start_task(cls, task_id: str, message: str = '任务开始执行'):
        with cls._lock:
            if task_id in cls._tasks:
                cls._tasks[task_id]['status'] = 1
                cls._tasks[task_id]['message'] = message
                _save_tasks_to_disk(cls._tasks)

    @classmethod
    def complete_task(cls, task_id: str, output: Optional[str] = None,
                      result_json: Optional[dict] = None, message: str = '任务完成'):
        with cls._lock:
            if task_id in cls._tasks:
                cls._tasks[task_id]['status'] = 2
                cls._tasks[task_id]['message'] = message
                cls._tasks[task_id]['result_file'] = output
                cls._tasks[task_id]['result_json'] = result_json
                cls._tasks[task_id]['error'] = None
                _save_tasks_to_disk(cls._tasks)

    @classmethod
    def fail_task(cls, task_id: str, error: str, message: str = '任务失败'):
        with cls._lock:
            if task_id in cls._tasks:
                cls._tasks[task_id]['status'] = 3
                cls._tasks[task_id]['message'] = message
                cls._tasks[task_id]['error'] = error
                _save_tasks_to_disk(cls._tasks)

    @classmethod
    def get_task(cls, task_id: str) -> Optional[Dict[str, Any]]:
        with cls._lock:
            return cls._tasks.get(task_id)