#!/usr/bin/env bash
# ==============================================================================
# nanoVLM x Google Colab CLI 一键工作流调度脚本
#
# 常用方式（在 /Users/mac/code/nanoVLM 根目录下执行）：
#   1. 首次开机 + 装依赖 + 传代码 + 预热模型与数据集（常驻显存）：
#      ./colab/run_colab.sh setup
#
#   2. 本地改完超参或逻辑后，0 等待直接复用显存模型开训（可反复执行）：
#      ./colab/run_colab.sh train
#      MAX_STEPS=500 LR_MP=0.002 ./colab/run_colab.sh train
#
#   3. 如果修改了本地 models/ 或 data/ 下的底层模块，秒级增量同步到云端：
#      ./colab/run_colab.sh sync
#
#   4. 拉回云端训练好的最优权重与实验 Notebook 记录到本地：
#      ./colab/run_colab.sh pull
#
#   5. 训练结束关闭云端 GPU 释放计费：
#      ./colab/run_colab.sh stop
#
#   6. 全自动无人值守跑完一条龙（setup -> train -> pull -> 自动 stop 关机）：
#      GPU=L4 MAX_STEPS=1000 ./colab/run_colab.sh all
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ------------------------ 可通过环境变量覆盖的全局配置 ------------------------
SESSION="${SESSION:-nanovlm}"
GPU="${GPU:-T4}"                       # 可选: T4, L4, G4, A100, H100
HIGH_MEM="${HIGH_MEM:-0}"              # 设为 1 开启 --high-mem (需 Pro/Pro+)
AUTH="${AUTH:-adc}"                    # 可选: adc 或 oauth2
SETUP_TIMEOUT="${SETUP_TIMEOUT:-1800}" # setup 超时默认 30 分钟
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-14400}" # train 超时默认 4 小时（最高建议 43200/86400）

COLAB_CMD=(colab --auth "$AUTH")

# macOS 防休眠包装（防止长时间训练时合盖/息屏断掉 WebSocket）
CAFFEINATE=()
if command -v caffeinate >/dev/null 2>&1; then
    CAFFEINATE=(caffeinate -i)
fi

build_env_flags() {
    ENV_FLAGS=()
    local vars=(
        HF_TOKEN HF_REPO_NAME WANDB_API_KEY WANDB_ENTITY LOG_WANDB
        VIT_MODEL_TYPE LM_MODEL_TYPE LM_TOKENIZER
        DATASET_PATH DATASET_NAMES STREAM_DATASET VAL_SIZE DATA_CUTOFF
        MAX_IMG_SIZE LM_MAX_LENGTH
        BATCH_SIZE GRAD_ACCUM MAX_STEPS EVAL_INTERVAL LOG_INTERVAL
        LR_MP LR_VISION LR_LANGUAGE RESET_WEIGHTS NUM_WORKERS
    )
    for var_name in "${vars[@]}"; do
        if [[ -n "${!var_name:-}" ]]; then
            ENV_FLAGS+=(--env "${var_name}=${!var_name}")
        fi
    done
}

sync_code() {
    echo "[run_colab] 正在打包本地源码 (models, data, utils, colab)..."
    local tmp_tar="/tmp/nanovlm_src_${SESSION}.tar.gz"
    COPYFILE_DISABLE=1 tar -czf "$tmp_tar" \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        models data utils colab generate.py pyproject.toml requirements.txt
    "${COLAB_CMD[@]}" upload -s "$SESSION" "$tmp_tar" /content/nanovlm_src.tar.gz
    rm -f "$tmp_tar"
    echo "[run_colab] 源码已同步至云端 /content/nanovlm_src.tar.gz"
}

cmd_setup() {
    echo "[run_colab] 1/4 检查或创建云端会话 '${SESSION}' (GPU=${GPU})..."
    if "${COLAB_CMD[@]}" status -s "$SESSION" >/dev/null 2>&1; then
        echo "[run_colab] 会话 '${SESSION}' 已存在，直接复用。"
    else
        local new_args=(-s "$SESSION" --gpu "$GPU")
        [[ "$HIGH_MEM" == "1" ]] && new_args+=(--high-mem)
        "${COLAB_CMD[@]}" new "${new_args[@]}"
    fi

    echo "[run_colab] 2/4 安装项目 Python 依赖 (requirements.txt + matplotlib)..."
    "${COLAB_CMD[@]}" install -s "$SESSION" -r requirements.txt matplotlib

    echo "[run_colab] 3/4 同步本地 nanoVLM 项目代码到云端..."
    sync_code

    echo "[run_colab] 4/4 执行 colab/setup.py 拉取模型与数据集并常驻显存..."
    build_env_flags
    "${CAFFEINATE[@]}" "${COLAB_CMD[@]}" exec -s "$SESSION" \
        --timeout "$SETUP_TIMEOUT" \
        "${ENV_FLAGS[@]+"${ENV_FLAGS[@]}"}" \
        -f colab/setup.py
}

cmd_train() {
    mkdir -p logs
    local plot_path="logs/loss_curve_$(date +%m%d_%H%M%S).png"
    echo "[run_colab] 启动增量训练 colab/train.py (timeout=${TRAIN_TIMEOUT}s)..."
    build_env_flags
    "${CAFFEINATE[@]}" "${COLAB_CMD[@]}" exec -s "$SESSION" \
        --timeout "$TRAIN_TIMEOUT" \
        --output-image "$plot_path" \
        "${ENV_FLAGS[@]+"${ENV_FLAGS[@]}"}" \
        -f colab/train.py

    if [[ -f "$plot_path" ]]; then
        echo "[run_colab] Loss 曲线已自动保存至本地: $plot_path"
    fi
}

cmd_pull() {
    mkdir -p checkpoints logs
    local local_tar="checkpoints/colab_best_${SESSION}_$(date +%m%d_%H%M%S).tar.gz"
    local log_nb="logs/colab_history_${SESSION}.ipynb"

    echo "[run_colab] 正在打包云端最优 Checkpoint 并拉回本地..."
    "${COLAB_CMD[@]}" exec -s "$SESSION" --timeout 300 <<'PY'
import os, tarfile
ckpt_dir = "/content/nanoVLM/checkpoints/latest_best"
out_tar = "/content/latest_best_ckpt.tar.gz"
if os.path.isdir(ckpt_dir):
    with tarfile.open(out_tar, "w:gz") as tar:
        tar.add(ckpt_dir, arcname="latest_best")
    print(f"[remote] 已打包 {ckpt_dir} -> {out_tar}")
else:
    raise FileNotFoundError(f"未找到云端权重目录: {ckpt_dir}")
PY
    "${COLAB_CMD[@]}" download -s "$SESSION" /content/latest_best_ckpt.tar.gz "$local_tar"
    tar -xzf "$local_tar" -C checkpoints/
    echo "[run_colab] 最优模型已解压至本地: checkpoints/latest_best/ (归档包: $local_tar)"

    "${COLAB_CMD[@]}" log -s "$SESSION" -o "$log_nb"
    echo "[run_colab] 完整云端执行历史已导出为 Notebook: $log_nb"
}

cmd_status() {
    "${COLAB_CMD[@]}" sessions
    "${COLAB_CMD[@]}" usage || true
    if "${COLAB_CMD[@]}" status -s "$SESSION" >/dev/null 2>&1; then
        "${COLAB_CMD[@]}" exec -s "$SESSION" <<'PY'
import subprocess
print(subprocess.check_output(["nvidia-smi"]).decode())
PY
    fi
}

cmd_stop() {
    echo "[run_colab] 正在停止并释放云端会话 '${SESSION}'..."
    "${COLAB_CMD[@]}" stop -s "$SESSION"
}

cmd_all() {
    trap 'echo "[run_colab] 触发退出清理，正在自动释放云端 GPU..."; cmd_stop || true' EXIT
    cmd_setup
    cmd_train
    cmd_pull
}

usage_help() {
    sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

case "${1:-help}" in
    setup)  cmd_setup ;;
    sync)   sync_code ;;
    train)  cmd_train ;;
    pull)   cmd_pull ;;
    status) cmd_status ;;
    stop)   cmd_stop ;;
    all)    cmd_all ;;
    *)      usage_help ;;
esac
