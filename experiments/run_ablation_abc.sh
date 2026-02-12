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
CUDA_DEVICES="${CUDA_DEVICES:-0}"
NUM_FRAMES="${NUM_FRAMES:-81}"
NUM_PROMPTS="${NUM_PROMPTS:-25}"

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

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python inference.py \
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

echo "=== Exp-A: sink_size ablation ==="
# Keep budget/recent fixed, vary sink via config.
run_case "Exp-A" "sink_06" "configs/self_forcing_dmd/self_forcing_dmd_sink6.yaml" 16 4
run_case "Exp-A" "sink_10" "configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml" 16 4
run_case "Exp-A" "sink_14" "configs/self_forcing_dmd/self_forcing_dmd_sink14.yaml" 16 4

echo "=== Exp-B: budget ablation ==="
# Keep sink/recent fixed, vary budget.
run_case "Exp-B" "budget_14" "configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml" 14 4
run_case "Exp-B" "budget_16" "configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml" 16 4
run_case "Exp-B" "budget_18" "configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml" 18 4

echo "=== Exp-C: recent ablation ==="
# Keep sink/budget fixed, vary recent.
run_case "Exp-C" "recent_02" "configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml" 16 2
run_case "Exp-C" "recent_04" "configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml" 16 4
run_case "Exp-C" "recent_06" "configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml" 16 6

echo "All experiments finished."
