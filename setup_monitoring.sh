#!/bin/bash

# 定时监控任务设置说明

echo "========================================"
echo "Deep Forcing 自动监控配置"
echo "========================================"

echo -e "\n方法1: 使用crontab设置定时任务"
echo "----------------------------------------"
echo "1. 打开crontab编辑器:"
echo "   crontab -e"
echo ""
echo "2. 添加以下行（每2小时检查一次）:"
echo "   0 */2 * * * /workspace/Projects/DeepForcing-fork/check_updates.sh >> /workspace/Projects/DeepForcing-fork/cron.log 2>&1"
echo ""
echo "3. 或者每天早上9点检查一次:"
echo "   0 9 * * * /workspace/Projects/DeepForcing-fork/check_updates.sh >> /workspace/Projects/DeepForcing-fork/cron.log 2>&1"
echo ""

echo -e "\n方法2: 创建systemd定时器（推荐用于Docker）"
echo "----------------------------------------"
echo "在Docker容器中，cron可能不可用，可以使用while循环:"
echo ""
echo "创建后台监控脚本:"
cat > /workspace/Projects/DeepForcing-fork/monitor_loop.sh << 'EOF'
#!/bin/bash
# 持续监控脚本
REPO_DIR="/workspace/Projects/DeepForcing-fork"
INTERVAL=7200  # 2小时（秒）

while true; do
    bash ${REPO_DIR}/check_updates.sh
    sleep ${INTERVAL}
done
EOF

chmod +x /workspace/Projects/DeepForcing-fork/monitor_loop.sh

echo "已创建 monitor_loop.sh"
echo ""
echo "使用方法:"
echo "  # 在后台运行监控（每2小时检查一次）"
echo "  nohup /workspace/Projects/DeepForcing-fork/monitor_loop.sh > /workspace/Projects/DeepForcing-fork/monitor.log 2>&1 &"
echo ""
echo "  # 查看监控日志"
echo "  tail -f /workspace/Projects/DeepForcing-fork/monitor.log"
echo ""
echo "  # 停止监控"
echo "  pkill -f monitor_loop.sh"

echo -e "\n方法3: 手动检查"
echo "----------------------------------------"
echo "随时运行以下命令手动检查更新:"
echo "  bash /workspace/Projects/DeepForcing-fork/check_updates.sh"

echo -e "\n========================================"
echo "配置完成！"
echo "========================================"
