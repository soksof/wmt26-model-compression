#!/usr/bin/env bash
# WMT26 model-compression — official inference entry point (vLLM backend).
#
# Contract forms (both work; argv is forwarded to the modelzip arg parser):
#   bash run.sh --lang-pair ces-deu --batch-size 8 --input in.txt --output out.txt
#   bash run.sh ces-deu 8 < in.txt > out.txt
#
# One translation per input line -> --output / stdout. All logs + progress -> stderr.
# Honours the caller's CUDA_VISIBLE_DEVICES.
set -euo pipefail
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Model: defaults to workdir/model; override with MODEL_DIR (or download it in setup.sh).
export MODEL_DIR="${MODEL_DIR:-$root_dir/workdir/model}"
# vLLM gotcha from our runs: the FlashInfer sampler's JIT can fail engine init -> disable.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# First-run torch.compile of the full model can exceed vLLM's 600s engine-ready default
# on slow CPUs (e.g. the GB10's ARM). Bump it; harmless when loading is fast.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"

# Per-package overrides, baked into the artifact (not the caller's shell) so the
# organizers' run reproduces it. The en-ar MBR contrastive ships a submission.env
# with `export MODELZIP_MBR=16`; the greedy PRIMARY package has no such file, so
# this line is a no-op there and run.sh stays byte-identical greedy.
if [ -f "$root_dir/submission.env" ]; then
  # shellcheck disable=SC1091
  . "$root_dir/submission.env"
fi

source "$root_dir/.venv/bin/activate"
# Exec the venv's interpreter by ABSOLUTE PATH, not PATH-resolved `python3`: the caller
# (e.g. modelzip.evaluate, or the organizers' harness) may invoke run.sh with an env
# where `python3` isn't this venv's — then inference.py imports the framework from the
# wrong place and dies with ModuleNotFoundError: modelzip. The absolute path is immune.
exec "$root_dir/.venv/bin/python" "$root_dir/inference.py" --model "$MODEL_DIR" "$@"
