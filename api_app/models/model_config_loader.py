from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CONFIG_DIR.parent.parent


class ModelConfigLoader:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = str(CONFIG_DIR / "model_config.json")
        self.config_path = config_path
        self._config = None
        self._load()

    def _load(self):
        for encoding in ("utf-8", "utf-8-sig", "gbk"):
            try:
                with open(self.config_path, encoding=encoding) as f:
                    self._config = json.load(f)
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        if self._config is None:
            raise RuntimeError(f"无法加载模型配置: {self.config_path}")

        project_root = str(PROJECT_DIR)
        for model in self._config.get("models", []):
            for key in ("yolo_path", "path"):
                if key in model:
                    p = model[key]
                    if os.path.isabs(p):
                        if not os.path.exists(p):
                            basename = os.path.basename(p)
                            candidate = os.path.join(project_root, "runs", basename)
                            if os.path.exists(candidate):
                                model[key] = candidate
                                logger.info(f"绝对路径 {p} 不存在，回退到 {candidate}")
                    else:
                        resolved = os.path.normpath(os.path.join(project_root, p))
                        model[key] = resolved

    def get_default_model(self) -> dict | None:
        models = self._config.get("models", [])
        for m in models:
            if m.get("type") == "yolo_seg":
                return m
        if models:
            return models[0]
        return None

    def get_model_by_id(self, model_id: str) -> dict | None:
        for m in self._config.get("models", []):
            if m.get("id") == model_id:
                return m
        return None

    def get_yolo_seg_model_path(self) -> str | None:
        m = self.get_default_model()
        if m:
            return m.get("yolo_path") or m.get("path")
        return None

    def list_models(self) -> list[dict]:
        return self._config.get("models", [])
