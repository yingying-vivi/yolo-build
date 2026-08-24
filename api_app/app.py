import os
import sys
from pathlib import Path

os.environ["YOLO_OFFLINE"] = "True"
os.environ["YOLO_AUTOINSTALL"] = "False"

_proj_dirs = [
    "/root/anaconda3/envs/torch310/share/proj",
    "/usr/share/proj",
    "/sys_old/usr/local/share/proj",
    str(Path(__file__).resolve().parent.parent / "proj_data"),
]
for d in _proj_dirs:
    if os.path.isdir(d) and os.path.exists(os.path.join(d, "proj.db")):
        os.environ["PROJ_LIB"] = d
        break

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask
from flask_cors import CORS

from .config import Config
from .routes.change_routes import image_bp


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    CORS(app, supports_credentials=True)

    Config.ensure_dirs()
    Config.init_app(app)

    app.register_blueprint(image_bp)

    return app


if __name__ == "__main__":
    app = create_app()
    app.logger.info(f"启动违建变化检测API服务，端口: {Config.PORT}")
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        use_reloader=False,
    )
