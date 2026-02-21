#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run_k6_deep_forcing.sh [options]

Options:
  --gpu <id>                 GPU id for CUDA_VISIBLE_DEVICES (default: 0)
  --output-dir <path>        Output directory (default: videos/k6_deep_forcing)
  --checkpoint <path>        Generator checkpoint (default: checkpoints/self_forcing_dmd.pt)
  --config <path>            Inference config (default: configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml)
  --prompt-src <path>        Prompt source txt (default: prompts/MovieGenVideoBench_extended.txt)
  --num-prompts <int>        Number of prompts from head of prompt-src (default: 25)
  --num-output-frames <int>  Number of output latent frames (default: 120)
  --overwrite                Remove existing video_*.mp4 and rerun all indices
  -h, --help                 Show this help
USAGE
}

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

GPU="0"
OUTPUT_DIR="videos/k6_deep_forcing"
CHECKPOINT="checkpoints/self_forcing_dmd.pt"
CONFIG="configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml"
PROMPT_SRC="prompts/MovieGenVideoBench_extended.txt"
NUM_PROMPTS=25
NUM_OUTPUT_FRAMES=120
OVERWRITE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --prompt-src)
      PROMPT_SRC="$2"
      shift 2
      ;;
    --num-prompts)
      NUM_PROMPTS="$2"
      shift 2
      ;;
    --num-output-frames)
      NUM_OUTPUT_FRAMES="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v python >/dev/null 2>&1; then
  echo "python not found in PATH"
  exit 1
fi

if ! [[ "$NUM_PROMPTS" =~ ^[0-9]+$ ]] || ! [[ "$NUM_OUTPUT_FRAMES" =~ ^[0-9]+$ ]]; then
  echo "--num-prompts and --num-output-frames must be non-negative integers"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/run.log"

# Stream both to console and log file.
exec > >(tee -a "$LOG_FILE") 2>&1

log "Starting k6 deep forcing batch run"
log "repo_root=${REPO_ROOT}"
log "gpu=${GPU} output_dir=${OUTPUT_DIR}"
log "config=${CONFIG} checkpoint=${CHECKPOINT}"
log "prompt_src=${PROMPT_SRC} num_prompts=${NUM_PROMPTS} num_output_frames=${NUM_OUTPUT_FRAMES}"

for path in "inference.py" "$CONFIG" "$CHECKPOINT" "$PROMPT_SRC"; do
  if [[ ! -f "$path" ]]; then
    log "ERROR: required file not found: $path"
    exit 1
  fi
done

PROMPTS_FILE="$OUTPUT_DIR/prompts_first${NUM_PROMPTS}.txt"
CSV_FILE="$OUTPUT_DIR/prompts.csv"
FAILED_FILE="$OUTPUT_DIR/failed_indices.txt"
TMP_PROMPT_FILE="$OUTPUT_DIR/.tmp_single_prompt.txt"

if [[ "$OVERWRITE" -eq 1 ]]; then
  log "Overwrite enabled: removing existing video_*.mp4"
  rm -f "$OUTPUT_DIR"/video_*.mp4 "$OUTPUT_DIR"/0-0_ema.mp4 "$FAILED_FILE"
fi

head -n "$NUM_PROMPTS" "$PROMPT_SRC" > "$PROMPTS_FILE"
RUN_PROMPTS="$(wc -l < "$PROMPTS_FILE" | tr -d ' ')"
if [[ "$RUN_PROMPTS" -eq 0 ]]; then
  log "ERROR: no prompts found in $PROMPT_SRC"
  exit 1
fi
if [[ "$RUN_PROMPTS" -lt "$NUM_PROMPTS" ]]; then
  log "WARN: only ${RUN_PROMPTS} prompts available (requested ${NUM_PROMPTS})"
fi

{
  echo "index,prompt"
  awk '{
    gsub(/"/, "\"\"", $0)
    printf "%d,\"%s\"\n", NR - 1, $0
  }' "$PROMPTS_FILE"
} > "$CSV_FILE"

log "Prepared prompt files: $PROMPTS_FILE and $CSV_FILE"

mapfile -t PROMPTS < "$PROMPTS_FILE"
rm -f "$FAILED_FILE"

for ((i = 0; i < RUN_PROMPTS; i++)); do
  target_video="$(printf "%s/video_%03d.mp4" "$OUTPUT_DIR" "$i")"

  if [[ "$OVERWRITE" -eq 0 && -f "$target_video" ]]; then
    log "[$((i + 1))/$RUN_PROMPTS] skip existing $(basename "$target_video")"
    continue
  fi

  printf '%s\n' "${PROMPTS[$i]}" > "$TMP_PROMPT_FILE"
  rm -f "$OUTPUT_DIR/0-0_ema.mp4"

  log "[$((i + 1))/$RUN_PROMPTS] generating $(basename "$target_video")"
  if CUDA_VISIBLE_DEVICES="$GPU" python inference.py \
    --config_path "$CONFIG" \
    --output_folder "$OUTPUT_DIR" \
    --checkpoint_path "$CHECKPOINT" \
    --data_path "$TMP_PROMPT_FILE" \
    --num_output_frames "$NUM_OUTPUT_FRAMES" \
    --num_samples 1 \
    --use_ema \
    --save_with_index
  then
    if [[ -f "$OUTPUT_DIR/0-0_ema.mp4" ]]; then
      mv -f "$OUTPUT_DIR/0-0_ema.mp4" "$target_video"
      log "[$((i + 1))/$RUN_PROMPTS] done $(basename "$target_video")"
    else
      log "ERROR: expected output not found for index $i"
      printf '%d\n' "$i" >> "$FAILED_FILE"
    fi
  else
    log "ERROR: inference failed for index $i"
    printf '%d\n' "$i" >> "$FAILED_FILE"
  fi
done

rm -f "$TMP_PROMPT_FILE" "$OUTPUT_DIR/0-0_ema.mp4"

generated_count="$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name 'video_*.mp4' | wc -l | tr -d ' ')"
log "Generated videos: ${generated_count}/${RUN_PROMPTS}"

if [[ -s "$FAILED_FILE" ]]; then
  log "FAILED indices: $(tr '\n' ' ' < "$FAILED_FILE")"
  exit 1
fi
rm -f "$FAILED_FILE"

if [[ "$generated_count" -ne "$RUN_PROMPTS" ]]; then
  log "ERROR: expected ${RUN_PROMPTS} videos, found ${generated_count}"
  exit 1
fi

log "Batch run completed successfully"
