#!/usr/bin/env bash
set -euo pipefail

# D group:
#   sink in [0..5], keep top_c=2 and recent=4
#   => budget = sink + recent + top_c = sink + 6
#
# E group:
#   sink in [0..5], keep top_c=0 and recent=4
#   => budget = sink + recent = sink + 4
#
# Outputs:
#   videos/Exp-D, videos/Exp-E
#   each experiment root contains prompts.csv
#   each case contains video_000.mp4 ... video_024.mp4

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/self_forcing_dmd.pt}"
NUM_FRAMES="${NUM_FRAMES:-81}"
NUM_PROMPTS="${NUM_PROMPTS:-25}"
ROUND="${ROUND:-ALL}"  # ALL | D | E

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
        matches[int(m.group(1))] = path

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
  local sink="$3"
  local budget="$4"
  local recent="$5"

  local config_path="configs/self_forcing_dmd/self_forcing_dmd_sink${sink}.yaml"
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

run_group_d() {
  echo "=== Exp-D: sink [0..5], top_c=2, recent=4 ==="

  # Round D-1 (4 GPUs)
  CUDA_VISIBLE_DEVICES=0 run_case "Exp-D" "sink_00_topc2_recent4" "0" "6" "4" &
  CUDA_VISIBLE_DEVICES=1 run_case "Exp-D" "sink_01_topc2_recent4" "1" "7" "4" &
  CUDA_VISIBLE_DEVICES=2 run_case "Exp-D" "sink_02_topc2_recent4" "2" "8" "4" &
  CUDA_VISIBLE_DEVICES=3 run_case "Exp-D" "sink_03_topc2_recent4" "3" "9" "4" &
  wait

  # Round D-2 (remaining 2 cases)
  CUDA_VISIBLE_DEVICES=0 run_case "Exp-D" "sink_04_topc2_recent4" "4" "10" "4" &
  CUDA_VISIBLE_DEVICES=1 run_case "Exp-D" "sink_05_topc2_recent4" "5" "11" "4" &
  wait
}

run_group_e() {
  echo "=== Exp-E: sink [0..5], top_c=0, recent=4 ==="

  # Round E-1 (4 GPUs)
  CUDA_VISIBLE_DEVICES=0 run_case "Exp-E" "sink_00_topc0_recent4" "0" "4" "4" &
  CUDA_VISIBLE_DEVICES=1 run_case "Exp-E" "sink_01_topc0_recent4" "1" "5" "4" &
  CUDA_VISIBLE_DEVICES=2 run_case "Exp-E" "sink_02_topc0_recent4" "2" "6" "4" &
  CUDA_VISIBLE_DEVICES=3 run_case "Exp-E" "sink_03_topc0_recent4" "3" "7" "4" &
  wait

  # Round E-2 (remaining 2 cases)
  CUDA_VISIBLE_DEVICES=0 run_case "Exp-E" "sink_04_topc0_recent4" "4" "8" "4" &
  CUDA_VISIBLE_DEVICES=1 run_case "Exp-E" "sink_05_topc0_recent4" "5" "9" "4" &
  wait
}

case "$ROUND" in
  ALL|all)
    run_group_d
    run_group_e
    ;;
  D|d|Exp-D|exp-d)
    run_group_d
    ;;
  E|e|Exp-E|exp-e)
    run_group_e
    ;;
  *)
    echo "Invalid ROUND=$ROUND. Use ROUND=ALL|D|E" >&2
    exit 1
    ;;
esac

echo "Done."
