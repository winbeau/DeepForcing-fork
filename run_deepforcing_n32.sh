#!/bin/bash
# Backward-compatible wrapper for the renamed DeepForcing batch script.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_deepforcing.sh" "$@"
