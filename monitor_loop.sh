#!/bin/bash
# 持续监控脚本
REPO_DIR="/workspace/Projects/DeepForcing-fork"
INTERVAL=7200  # 2小时（秒）

while true; do
    bash ${REPO_DIR}/check_updates.sh
    sleep ${INTERVAL}
done
