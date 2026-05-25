#!/bin/bash
set -e

PROJECT="/home/fumu/PycharmProjects/ultralytics-main"
DEPLOY_DIR="/tmp/building_detect_deploy"
PKG_NAME="building_detect_deploy.tar.gz"

echo "=== 清理旧打包 ==="
rm -rf "$DEPLOY_DIR" "$PKG_NAME"

echo "=== 创建部署目录结构 ==="
mkdir -p "$DEPLOY_DIR/runs/building_seg_v5/weights"
mkdir -p "$DEPLOY_DIR/api_app/models"
mkdir -p "$DEPLOY_DIR/api_app/routes"
mkdir -p "$DEPLOY_DIR/api_app/services"
mkdir -p "$DEPLOY_DIR/api_app/utils"
mkdir -p "$DEPLOY_DIR/ultralytics"

echo "=== 复制权重文件 ==="
cp "$PROJECT/runs/building_seg_v5/weights/best.pt" "$DEPLOY_DIR/runs/building_seg_v5/weights/"

echo "=== 复制 api_app ==="
cp "$PROJECT/api_app/__init__.py" "$DEPLOY_DIR/api_app/"
cp "$PROJECT/api_app/app.py" "$DEPLOY_DIR/api_app/"
cp "$PROJECT/api_app/config.py" "$DEPLOY_DIR/api_app/"
cp "$PROJECT/api_app/requirements.txt" "$DEPLOY_DIR/api_app/"
cp "$PROJECT/api_app/models/__init__.py" "$DEPLOY_DIR/api_app/models/"
cp "$PROJECT/api_app/models/model_config.json" "$DEPLOY_DIR/api_app/models/"
cp "$PROJECT/api_app/models/model_config_loader.py" "$DEPLOY_DIR/api_app/models/"
cp "$PROJECT/api_app/models/model_wrapper.py" "$DEPLOY_DIR/api_app/models/"
cp "$PROJECT/api_app/routes/__init__.py" "$DEPLOY_DIR/api_app/routes/"
cp "$PROJECT/api_app/routes/change_routes.py" "$DEPLOY_DIR/api_app/routes/"
cp "$PROJECT/api_app/services/__init__.py" "$DEPLOY_DIR/api_app/services/"
cp "$PROJECT/api_app/services/change_service.py" "$DEPLOY_DIR/api_app/services/"
cp "$PROJECT/api_app/utils/__init__.py" "$DEPLOY_DIR/api_app/utils/"
cp "$PROJECT/api_app/utils/file_utils.py" "$DEPLOY_DIR/api_app/utils/"
cp "$PROJECT/api_app/utils/image_utils.py" "$DEPLOY_DIR/api_app/utils/"
cp "$PROJECT/api_app/utils/task_manager.py" "$DEPLOY_DIR/api_app/utils/"

echo "=== 复制修改后的 ultralytics ==="
cp -r "$PROJECT/ultralytics/__init__.py" "$DEPLOY_DIR/ultralytics/"
cp -r "$PROJECT/ultralytics/cfg" "$DEPLOY_DIR/ultralytics/"
cp -r "$PROJECT/ultralytics/data" "$DEPLOY_DIR/ultralytics/"
cp -r "$PROJECT/ultralytics/engine" "$DEPLOY_DIR/ultralytics/"
cp -r "$PROJECT/ultralytics/hub" "$DEPLOY_DIR/ultralytics/"
cp -r "$PROJECT/ultralytics/models" "$DEPLOY_DIR/ultralytics/"
cp -r "$PROJECT/ultralytics/nn" "$DEPLOY_DIR/ultralytics/"
cp -r "$PROJECT/ultralytics/optim" "$DEPLOY_DIR/ultralytics/"
cp -r "$PROJECT/ultralytics/solutions" "$DEPLOY_DIR/ultralytics/"
cp -r "$PROJECT/ultralytics/utils" "$DEPLOY_DIR/ultralytics/"

echo "=== 生成甲方 start.sh ==="
cat > "$DEPLOY_DIR/start.sh" << 'STARTSH'
#!/bin/bash
CONDA_PYTHON="/root/anaconda3/envs/torch310/bin/python"
PROJECT_DIR="/home/PycharmProjects/ultralytics-main"

export PYTHONPATH="$PROJECT_DIR"
export YOLO_OFFLINE=True
export YOLO_AUTOINSTALL=False

cd "$PROJECT_DIR" || { echo "无法切换到目录：$PROJECT_DIR"; exit 1; }

mkdir -p "$PROJECT_DIR/api_app/logs"

echo "启动违建变化检测API服务..."
nohup $CONDA_PYTHON -m api_app.app > "$PROJECT_DIR/api_app/logs/startup.log" 2>&1 &
echo "服务PID: $!"
echo "日志: $PROJECT_DIR/api_app/logs/startup.log"
echo "端口: 9030"
STARTSH
chmod +x "$DEPLOY_DIR/start.sh"

echo "=== 打包 ==="
cd /tmp
tar -czf "$PROJECT/$PKG_NAME" -C /tmp building_detect_deploy

echo "=== 完成 ==="
PKG_PATH="$PROJECT/$PKG_NAME"
PKG_SIZE=$(du -sh "$PKG_PATH" | cut -f1)
echo "部署包: $PKG_PATH ($PKG_SIZE)"
echo ""
echo "部署步骤:"
echo "  1. 通过Xterminal把 $PKG_PATH 传到甲方服务器"
echo "  2. 在甲方服务器上执行:"
echo "     cd /home/PycharmProjects/ultralytics-main"
echo "     tar -xzf building_detect_deploy.tar.gz --strip-components=1"
echo "     bash start.sh"