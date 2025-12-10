#!/bin/bash

# Deep Forcing 代码发布监控脚本
# 用途：自动检查GitHub仓库是否有代码更新

REPO_DIR="/workspace/Projects/DeepForcing-fork"
REPO_URL="https://github.com/cvlab-kaist/DeepForcing.git"
LOG_FILE="${REPO_DIR}/update_log.txt"

echo "========================================"
echo "Deep Forcing 代码监控脚本"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

cd "${REPO_DIR}" || exit 1

# 获取远程更新
git fetch origin > /dev/null 2>&1

# 检查是否有新的commit
LOCAL_HASH=$(git rev-parse HEAD)
REMOTE_HASH=$(git rev-parse origin/master)

if [ "$LOCAL_HASH" != "$REMOTE_HASH" ]; then
    echo "🎉 检测到新的更新！" | tee -a "${LOG_FILE}"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_FILE}"

    # 显示更新日志
    echo -e "\n📝 更新内容:" | tee -a "${LOG_FILE}"
    git log HEAD..origin/master --oneline | tee -a "${LOG_FILE}"

    # 拉取更新
    echo -e "\n⬇️  正在拉取更新..." | tee -a "${LOG_FILE}"
    git pull origin master

    # 检查是否有Python文件（代码发布的标志）
    if find . -name "*.py" -type f | grep -v ".git" | head -1 > /dev/null; then
        echo -e "\n✅ 发现Python代码文件！代码可能已经发布！" | tee -a "${LOG_FILE}"
        echo "请检查以下文件:" | tee -a "${LOG_FILE}"
        find . -name "*.py" -type f | grep -v ".git" | head -20 | tee -a "${LOG_FILE}"

        # 检查是否有requirements.txt
        if [ -f "requirements.txt" ]; then
            echo -e "\n📦 发现 requirements.txt，依赖清单：" | tee -a "${LOG_FILE}"
            cat requirements.txt | tee -a "${LOG_FILE}"
        fi

        # 检查是否有README更新
        if [ -f "README.md" ]; then
            echo -e "\n📖 README.md 内容：" | tee -a "${LOG_FILE}"
            head -50 README.md | tee -a "${LOG_FILE}"
        fi

        echo -e "\n🚀 建议立即查看仓库并开始复现！" | tee -a "${LOG_FILE}"
    else
        echo -e "\n⚠️  更新中暂未包含Python代码文件" | tee -a "${LOG_FILE}"
    fi

    # 发送通知（如果配置了）
    if command -v notify-send &> /dev/null; then
        notify-send "Deep Forcing更新" "检测到代码仓库有新的提交！"
    fi
else
    echo "📭 暂无更新 (最后检查: $(date '+%Y-%m-%d %H:%M:%S'))"
fi

echo -e "\n当前状态:"
echo "- 本地commit: $LOCAL_HASH"
echo "- 远程commit: $REMOTE_HASH"
echo "- Python文件数: $(find . -name "*.py" -type f | grep -v ".git" | wc -l)"
echo "========================================"
