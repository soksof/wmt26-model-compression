#!/usr/bin/env bash
# Prepare an inference-only environment (vLLM). Run once before run.sh.
# Mirrors the modelzip baseline's setup.sh; only requirements.txt differs (vLLM stack).
set -euo pipefail
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
venv_dir="$root_dir/.venv"
# Editable install of the shared modelzip framework (two levels up = repo root when this
# dir lives at submissions/<id>/ inside the organizers' repo).
modelzip_source="${MODELZIP_SOURCE:-$(cd "$root_dir/../.." && pwd)}"

uv venv --python 3.12 "$venv_dir"
source "$venv_dir/bin/activate"
uv pip install -r "$root_dir/requirements.txt"
uv pip install --no-deps -e "$modelzip_source"

# Model: bundle it at workdir/model, OR uncomment to pull from a HF repo link:
# huggingface-cli download <org>/<repo-id> --local-dir "$root_dir/workdir/model"

hf download soksof/gemma-3-12b-wmt26-vocaball-int4 --local-dir "$root_dir/workdir/model"
