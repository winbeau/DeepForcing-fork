#!/bin/bash
# Multi-GPU parallel inference for DeepForcing experiments.
set -e

# ============ Defaults ============
OUTPUT_DIR=""
PROMPT_FILE="./prompts/MovieGenVideoBench_num32.txt"
CONFIG_PATH="configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml"
CHECKPOINT_PATH="checkpoints/self_forcing_dmd.pt"
NUM_FRAMES=120
NUM_GPUS=4
SEED=1356145
PROFILE=0
PYTHON_CMD=(uv run python)
# ==================================

usage() {
    exit_code="${1:-1}"
    cat <<EOF
Usage: bash run_deepforcing.sh --output_dir <dir> [options]

Required:
  --output_dir DIR        Output directory

Options:
  --num_gpus N            Number of GPUs (default: $NUM_GPUS)
  --num-frames N          Number of output frames (default: $NUM_FRAMES)
  --config_path PATH      Config file (default: $CONFIG_PATH)
  --checkpoint_path PATH  Checkpoint file (default: $CHECKPOINT_PATH)
  --prompt_file PATH      Prompt file (default: $PROMPT_FILE)
  --seed N                Random seed (default: $SEED)
  --profile               Print DiT generation and VAE decoding timing

GPU assignment:
  - If CUDA_VISIBLE_DEVICES is unset, this script uses GPUs 0..N-1.
  - If CUDA_VISIBLE_DEVICES is set, its visible GPU count must equal --num_gpus.

Environment:
  - MASTER_PORT is passed through to child inference processes unchanged.
EOF
    exit "$exit_code"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir)        OUTPUT_DIR="$2"; shift 2 ;;
        --num_gpus)          NUM_GPUS="$2"; shift 2 ;;
        --num-frames)        NUM_FRAMES="$2"; shift 2 ;;
        --num_output_frames) NUM_FRAMES="$2"; shift 2 ;;
        --config_path)       CONFIG_PATH="$2"; shift 2 ;;
        --checkpoint_path)   CHECKPOINT_PATH="$2"; shift 2 ;;
        --prompt_file)       PROMPT_FILE="$2"; shift 2 ;;
        --seed)              SEED="$2"; shift 2 ;;
        --profile)           PROFILE=1; shift ;;
        -h|--help)           usage 0 ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

[ -z "$OUTPUT_DIR" ] && { echo "Error: --output_dir is required"; usage; }

PIDS=()
WORKER_INDICES=()
cleanup() {
    echo ""
    echo "Caught interrupt, killing all GPU processes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    wait || true
    exit 1
}
trap cleanup SIGINT SIGTERM

TOTAL_PROMPTS=$(python3 -c "
import sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    print(sum(1 for line in f if line.strip()))
" "$PROMPT_FILE")
if [ "$TOTAL_PROMPTS" -eq 0 ]; then
    echo "Error: no non-empty prompts found in ${PROMPT_FILE}"
    exit 1
fi
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
echo "Output frames: $NUM_FRAMES"
echo "Profile: $PROFILE"
echo "Python command: ${PYTHON_CMD[*]}"
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"
if [[ -n "${MASTER_PORT:-}" ]]; then
    echo "MASTER_PORT: $MASTER_PORT"
else
    echo "MASTER_PORT: <unset>"
fi
echo "Total prompts: $TOTAL_PROMPTS"
echo "Prompts per GPU: $PROMPTS_PER_GPU"
echo ""

mkdir -p "$OUTPUT_DIR"
rm -rf "${OUTPUT_DIR}/.tmp_gpu"* "${OUTPUT_DIR}/.prompts_gpu"*

PROFILE_ARGS=()
if [[ "$PROFILE" -eq 1 ]]; then
    PROFILE_ARGS+=(--profile)
fi

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
    python3 -c "
import sys
prompt_file, output_file = sys.argv[1], sys.argv[2]
start, end = int(sys.argv[3]), int(sys.argv[4])
with open(prompt_file, 'r', encoding='utf-8') as f:
    prompts = [line.strip() for line in f if line.strip()]
with open(output_file, 'w', encoding='utf-8') as f:
    for prompt in prompts[start - 1:end]:
        f.write(prompt + '\n')
" "$PROMPT_FILE" "$tmp_file" "$start_line" "$end_line"

    num_lines=$(python3 -c "
import sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    print(sum(1 for line in f if line.strip()))
" "$tmp_file")
    [ "$num_lines" -eq 0 ] && continue

    gpu_out="${OUTPUT_DIR}/.tmp_gpu${worker_idx}"
    mkdir -p "$gpu_out"
    worker_log="${OUTPUT_DIR}/worker_${worker_idx}.log"

    echo "[Worker $worker_idx -> GPU $physical_gpu] prompts $((start_line - 1))-$((end_line - 1))  ($num_lines prompts)"

    CUDA_VISIBLE_DEVICES=$physical_gpu "${PYTHON_CMD[@]}" inference.py \
        --config_path "$CONFIG_PATH" \
        --output_folder "$gpu_out" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --data_path "$tmp_file" \
        --num_output_frames "$NUM_FRAMES" \
        --use_ema \
        --save_with_index \
        --seed "$SEED" \
        "${PROFILE_ARGS[@]}" > >(tee "$worker_log") 2>&1 &
    PIDS+=($!)
    WORKER_INDICES+=("$worker_idx")
done

echo ""
echo "Waiting for all GPUs to finish..."
failed_workers=0
for pid_index in "${!PIDS[@]}"; do
    pid="${PIDS[$pid_index]}"
    worker_idx="${WORKER_INDICES[$pid_index]}"
    if ! wait "$pid"; then
        echo "ERROR: worker ${worker_idx} failed. See ${OUTPUT_DIR}/worker_${worker_idx}.log"
        failed_workers=$((failed_workers + 1))
    fi
done
echo "All GPUs finished!"

# ---------- Rename to video_XXX.mp4 ----------
echo "Renaming output files..."
missing_outputs=0
renamed_outputs=0
for worker_idx in $(seq 0 $((NUM_GPUS - 1))); do
    offset=$((worker_idx * PROMPTS_PER_GPU))
    gpu_out="${OUTPUT_DIR}/.tmp_gpu${worker_idx}"
    [ ! -d "$gpu_out" ] && continue

    for local_idx in $(seq 0 $((PROMPTS_PER_GPU - 1))); do
        src="${gpu_out}/${local_idx}-0_ema.mp4"
        global_idx=$((offset + local_idx))
        [ "$global_idx" -ge "$TOTAL_PROMPTS" ] && continue
        if [ ! -f "$src" ]; then
            echo "ERROR: missing output for prompt index ${global_idx}: ${src}"
            echo "       See ${OUTPUT_DIR}/worker_${worker_idx}.log"
            missing_outputs=$((missing_outputs + 1))
            continue
        fi
        dst="${OUTPUT_DIR}/video_$(printf '%03d' "$global_idx").mp4"
        mv -f "$src" "$dst"
        renamed_outputs=$((renamed_outputs + 1))
        echo "  ${dst}"
    done
done

if [ "$failed_workers" -ne 0 ] || [ "$missing_outputs" -ne 0 ]; then
    echo "ERROR: ${failed_workers} workers failed; ${missing_outputs} expected videos were not generated."
    echo "Temporary outputs and worker logs are preserved under ${OUTPUT_DIR}/"
    exit 1
fi

if [ "$renamed_outputs" -ne "$TOTAL_PROMPTS" ]; then
    echo "ERROR: expected ${TOTAL_PROMPTS} videos, renamed ${renamed_outputs}."
    echo "Temporary outputs and worker logs are preserved under ${OUTPUT_DIR}/"
    exit 1
fi

# ---------- Cleanup temp files ----------
rm -rf "${OUTPUT_DIR}/.tmp_gpu"* "${OUTPUT_DIR}/.prompts_gpu"*

echo ""
echo "=== Done ==="
echo "Output: ${OUTPUT_DIR}/"
echo "Videos: ${renamed_outputs} / ${TOTAL_PROMPTS}"
