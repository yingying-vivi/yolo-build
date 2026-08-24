import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent


class Config:
    HOST = "0.0.0.0"
    PORT = 9030
    DEBUG = True

    MAX_CONTENT_LENGTH = 10 * 1024 * 1024 * 1024

    UPLOAD_DIR = str(BASE_DIR / "static" / "uploads")
    RESULT_DIR = str(BASE_DIR / "static" / "results")
    TEMP_DIR = str(BASE_DIR / "static" / "temp")
    API_RESULTS_DIR = str(BASE_DIR / "api_results")
    LOG_DIR = str(BASE_DIR / "logs")

    DEFAULT_CONF_THRESHOLD = 0.3
    DEFAULT_IOU_THRESHOLD = 0.5
    DEFAULT_NMS_IOU_THRESHOLD = 0.5
    DEFAULT_MATCH_IOU_THRESHOLD = 0.5
    DEFAULT_AREA_CHANGE_THRESHOLD = 0.2
    DEFAULT_MIN_AREA_PIXELS = 500
    DEFAULT_CROP_SIZE = 4096
    DEFAULT_TILE_SIZE = 1024
    DEFAULT_STRIDE = 800
    DEFAULT_VIS_TILE_SIZE = 1024
    DEFAULT_TARGET_RES = 0.2

    THREAD_POOL_MAX_WORKERS = 1

    CLEANUP_INTERVAL = 3600
    MAX_TASK_AGE_HOURS = 24

    @staticmethod
    def ensure_dirs():
        dirs = [
            Config.UPLOAD_DIR,
            Config.RESULT_DIR,
            Config.TEMP_DIR,
            Config.API_RESULTS_DIR,
            Config.LOG_DIR,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    @staticmethod
    def init_app(app):
        import logging
        from logging.handlers import RotatingFileHandler

        log_dir = Config.LOG_DIR
        os.makedirs(log_dir, exist_ok=True)

        handler = RotatingFileHandler(
            os.path.join(log_dir, "api.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        app.logger.addHandler(handler)
        app.logger.setLevel(logging.INFO)
