#!/bin/bash
# Multi-GPU parallel inference — 32 prompts, 120 latent frames each
set -e

# ============ Defaults ============
OUTPUT_DIR=""
PROMPT_FILE="./prompts/MovieGenVideoBench_num32.txt"
CONFIG_PATH="configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml"
CHECKPOINT_PATH="checkpoints/self_forcing_dmd.pt"
NUM_OUTPUT_FRAMES=120
NUM_GPUS=4
SEED=1356145
# ==================================

usage() {
    cat <<EOF
Usage: bash run_deepforcing_n32.sh --output_dir <dir> [options]

Required:
  --output_dir DIR        Output directory

Options:
  --num_gpus N            Number of GPUs (default: 4)
  --config_path PATH      Config file (default: $CONFIG_PATH)
  --checkpoint_path PATH  Checkpoint file (default: $CHECKPOINT_PATH)
  --prompt_file PATH      Prompt file (default: $PROMPT_FILE)
  --num_output_frames N   Latent frames (default: $NUM_OUTPUT_FRAMES)
  --seed N                Random seed (default: $SEED)

GPU assignment:
  - If CUDA_VISIBLE_DEVICES is unset, this script uses GPUs 0..N-1.
  - If CUDA_VISIBLE_DEVICES is set, its visible GPU count must equal --num_gpus.
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir)        OUTPUT_DIR="$2"; shift 2 ;;
        --num_gpus)          NUM_GPUS="$2"; shift 2 ;;
        --config_path)       CONFIG_PATH="$2"; shift 2 ;;
        --checkpoint_path)   CHECKPOINT_PATH="$2"; shift 2 ;;
        --prompt_file)       PROMPT_FILE="$2"; shift 2 ;;
        --num_output_frames) NUM_OUTPUT_FRAMES="$2"; shift 2 ;;
        --seed)              SEED="$2"; shift 2 ;;
        -h|--help)           usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

[ -z "$OUTPUT_DIR" ] && { echo "Error: --output_dir is required"; usage; }

PIDS=()
cleanup() {
    echo ""
    echo "Caught interrupt, killing all GPU processes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    wait
    exit 1
}
trap cleanup SIGINT SIGTERM

TOTAL_PROMPTS=$(wc -l < "$PROMPT_FILE")
PROMPTS_PER_GPU=$(( (TOTAL_PROMPTS + NUM_GPUS - 1) / NUM_GPUS ))

GPU_ASSIGNMENT_SOURCE="default"
GPU_IDS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    visible_gpus=$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr -d '[:space:]')
    IFS=',' read -r -a GPU_IDS <<< "$visible_gpus"

    if [[ ${#GPU_IDS[@]} -ne "$NUM_GPUS" ]]; then
        echo "Error: CUDA_VISIBLE_DEVICES specifies ${#GPU_IDS[@]} GPUs (${visible_gpus}), but --num_gpus is ${NUM_GPUS}."
        exit 1
    fi

    for gpu_id in "${GPU_IDS[@]}"; do
        if [[ -z "$gpu_id" ]]; then
            echo "Error: CUDA_VISIBLE_DEVICES contains an empty GPU id."
            exit 1
        fi
    done

    GPU_ASSIGNMENT_SOURCE="env"
else
    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        GPU_IDS+=("$gpu_id")
    done
fi

echo "=== Multi-GPU Parallel Inference ==="
echo "GPUs: $NUM_GPUS"
echo "GPU assignment source: $GPU_ASSIGNMENT_SOURCE"
echo "Physical GPUs: ${GPU_IDS[*]}"
echo "Total prompts: $TOTAL_PROMPTS"
echo "Prompts per GPU: $PROMPTS_PER_GPU"
echo "Latent frames: $NUM_OUTPUT_FRAMES"
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# ---------- Generate prompts.csv ----------
python3 -c "
import csv, sys
prompt_file, output_csv = sys.argv[1], sys.argv[2]
with open(prompt_file, 'r', encoding='utf-8') as f:
    prompts = [line.strip() for line in f if line.strip()]
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['index', 'prompt'])
    for i, p in enumerate(prompts):
        writer.writerow([i, p])
print(f'Generated {output_csv} with {len(prompts)} prompts')
" "$PROMPT_FILE" "${OUTPUT_DIR}/prompts.csv"

# ---------- Launch per-GPU inference ----------
for worker_idx in $(seq 0 $((NUM_GPUS - 1))); do
    physical_gpu="${GPU_IDS[$worker_idx]}"
    start_line=$((worker_idx * PROMPTS_PER_GPU + 1))
    end_line=$(( (worker_idx + 1) * PROMPTS_PER_GPU ))
    [ "$end_line" -gt "$TOTAL_PROMPTS" ] && end_line=$TOTAL_PROMPTS

    # Extract this worker's subset of prompts.
    tmp_file="${OUTPUT_DIR}/.prompts_gpu${worker_idx}.txt"
    sed -n "${start_line},${end_line}p" "$PROMPT_FILE" > "$tmp_file"

    num_lines=$(wc -l < "$tmp_file")
    [ "$num_lines" -eq 0 ] && continue

    gpu_out="${OUTPUT_DIR}/.tmp_gpu${worker_idx}"
    mkdir -p "$gpu_out"

    echo "[Worker $worker_idx -> GPU $physical_gpu] prompts $((start_line - 1))-$((end_line - 1))  ($num_lines prompts)"

    CUDA_VISIBLE_DEVICES=$physical_gpu python inference.py \
        --config_path "$CONFIG_PATH" \
        --output_folder "$gpu_out" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --data_path "$tmp_file" \
        --num_output_frames "$NUM_OUTPUT_FRAMES" \
        --use_ema \
        --save_with_index \
        --seed "$SEED" &
    PIDS+=($!)
done

echo ""
echo "Waiting for all GPUs to finish..."
wait
echo "All GPUs finished!"

# ---------- Rename to video_XXX.mp4 ----------
echo "Renaming output files..."
for worker_idx in $(seq 0 $((NUM_GPUS - 1))); do
    offset=$((worker_idx * PROMPTS_PER_GPU))
    gpu_out="${OUTPUT_DIR}/.tmp_gpu${worker_idx}"
    [ ! -d "$gpu_out" ] && continue

    for local_idx in $(seq 0 $((PROMPTS_PER_GPU - 1))); do
        src="${gpu_out}/${local_idx}-0_ema.mp4"
        [ ! -f "$src" ] && continue
        global_idx=$((offset + local_idx))
        dst="video_$(printf '%03d' "$global_idx").mp4"
        mv "$src" "${OUTPUT_DIR}/${dst}"
        echo "  ${dst}"
    done
done

# ---------- Cleanup temp files ----------
rm -rf "${OUTPUT_DIR}/.tmp_gpu"* "${OUTPUT_DIR}/.prompts_gpu"*

echo ""
echo "=== Done ==="
echo "Output: ${OUTPUT_DIR}/"
echo "Videos: $(ls "${OUTPUT_DIR}"/video_*.mp4 2>/dev/null | wc -l) / ${TOTAL_PROMPTS}"
