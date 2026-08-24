#!/bin/bash
CONDA_PYTHON="/home/fumu/conda_disk/anaconda3/envs/wrj_torch/bin/python"
PROJECT_DIR="/home/fumu/PycharmProjects/ultralytics-main"

export PYTHONPATH="$PROJECT_DIR"
export YOLO_OFFLINE=True
export YOLO_AUTOINSTALL=False
export PROJ_LIB="/home/fumu/conda_disk/anaconda3/envs/wrj_torch/share/proj"

mkdir -p "$PROJECT_DIR/api_app/logs"

echo "启动违建变化检测API服务..."
nohup $CONDA_PYTHON -m api_app.app > "$PROJECT_DIR/api_app/logs/startup.log" 2>&1 &
echo "服务PID: $!"
echo "日志: $PROJECT_DIR/api_app/logs/startup.log"
echo "端口: 9030"
