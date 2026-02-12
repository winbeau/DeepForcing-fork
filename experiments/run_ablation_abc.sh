#!/usr/bin/env bash
set -euo pipefail

# Run three ablation groups:
# Exp-A: sink_size ablation
# Exp-B: budget ablation
# Exp-C: recent ablation
#
# Outputs are saved under:
# videos/Exp-A, videos/Exp-B, videos/Exp-C
# Each experiment root gets prompts.csv and each case saves:
# video_000.mp4 ... video_024.mp4

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/self_forcing_dmd.pt}"
NUM_FRAMES="${NUM_FRAMES:-81}"
NUM_PROMPTS="${NUM_PROMPTS:-25}"
ROUND="${ROUND:-ALL}"

PROMPT_SRC="prompts/MovieGenVideoBench_extended.txt"
PROMPT_TXT="experiments/prompts_25.txt"
PROMPT_CSV="experiments/prompts.csv"

python experiments/prepare_prompts.py \
  --src "$PROMPT_SRC" \
  --num "$NUM_PROMPTS" \
  --out-txt "$PROMPT_TXT" \
  --out-csv "$PROMPT_CSV"

rename_videos() {
  local out_dir="$1"
  local expected_count="$2"
  python - "$out_dir" "$expected_count" <<'PY'
import glob
import os
import re
import shutil
import sys

out_dir = sys.argv[1]
expected = int(sys.argv[2])

matches = {}
for path in glob.glob(os.path.join(out_dir, "*.mp4")):
    base = os.path.basename(path)
    m = re.match(r"^(\d+)-\d+_(?:ema|regular)\.mp4$", base)
    if m:
        idx = int(m.group(1))
        matches[idx] = path

missing = [i for i in range(expected) if i not in matches]
if missing:
    raise RuntimeError(f"Missing generated videos for prompt indices: {missing}")

for i in range(expected):
    src = matches[i]
    dst = os.path.join(out_dir, f"video_{i:03d}.mp4")
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.move(src, dst)
PY
}

run_case() {
  local exp_name="$1"
  local case_name="$2"
  local config_path="$3"
  local budget="$4"
  local recent="$5"

  local exp_root="videos/${exp_name}"
  local out_dir="${exp_root}/${case_name}"
  mkdir -p "$out_dir"
  cp "$PROMPT_CSV" "${exp_root}/prompts.csv"

  python inference.py \
    --config_path "$config_path" \
    --output_folder "$out_dir" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --data_path "$PROMPT_TXT" \
    --num_output_frames "$NUM_FRAMES" \
    --num_samples 1 \
    --use_ema \
    --save_with_index \
    --Budget "$budget" \
    --Recent "$recent"

  rename_videos "$out_dir" "$NUM_PROMPTS"
}

run_round_a() {
  echo "=== Exp-A: sink_size ablation (4 GPUs) ==="
  # Keep budget/recent fixed, vary sink via config.
  CUDA_VISIBLE_DEVICES=0 run_case "Exp-A" "sink_06" "configs/self_forcing_dmd/self_forcing_dmd_sink6.yaml" 16 4 &
  CUDA_VISIBLE_DEVICES=1 run_case "Exp-A" "sink_08" "configs/self_forcing_dmd/self_forcing_dmd_sink8.yaml" 16 4 &
  CUDA_VISIBLE_DEVICES=2 run_case "Exp-A" "sink_10" "configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml" 16 4 &
  CUDA_VISIBLE_DEVICES=3 run_case "Exp-A" "sink_12" "configs/self_forcing_dmd/self_forcing_dmd_sink12.yaml" 16 4 &
  wait
}

run_round_b() {
  echo "=== Exp-B: budget ablation (4 GPUs) ==="
  # Keep sink/recent fixed, vary budget.
  CUDA_VISIBLE_DEVICES=0 run_case "Exp-B" "budget_14" "configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml" 14 4 &
  CUDA_VISIBLE_DEVICES=1 run_case "Exp-B" "budget_15" "configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml" 15 4 &
  CUDA_VISIBLE_DEVICES=2 run_case "Exp-B" "budget_16" "configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml" 16 4 &
  CUDA_VISIBLE_DEVICES=3 run_case "Exp-B" "budget_18" "configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml" 18 4 &
  wait
}

run_round_c() {
  echo "=== Exp-C: recent ablation (4 GPUs) ==="
  # Keep sink/budget fixed, vary recent.
  CUDA_VISIBLE_DEVICES=0 run_case "Exp-C" "recent_02" "configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml" 16 2 &
  CUDA_VISIBLE_DEVICES=1 run_case "Exp-C" "recent_03" "configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml" 16 3 &
  CUDA_VISIBLE_DEVICES=2 run_case "Exp-C" "recent_04" "configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml" 16 4 &
  CUDA_VISIBLE_DEVICES=3 run_case "Exp-C" "recent_06" "configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml" 16 6 &
  wait
}

case "$ROUND" in
  ALL|all)
    run_round_a
    run_round_b
    run_round_c
    ;;
  A|a|Exp-A|exp-a)
    run_round_a
    ;;
  B|b|Exp-B|exp-b)
    run_round_b
    ;;
  C|c|Exp-C|exp-c)
    run_round_c
    ;;
  *)
    echo "Invalid ROUND=$ROUND. Use ROUND=ALL|A|B|C" >&2
    exit 1
    ;;
esac

echo "All experiments finished."
