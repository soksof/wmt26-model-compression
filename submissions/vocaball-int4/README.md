# `<system-id>` — WMT26 Model Compression submission (vLLM)

Compressed **Gemma 3 12B**, served with **vLLM**. Compression: `<fill: e.g. INT4 W4A16 /
FP8 / depth-36 contiguous prune + KD heal + no-Arabic vocab prune + INT4>`.

- **Directions (`--lang-pair`):** `<ces-deu, eng-zho_Hans / or eng-ara_EG>`
- **Model:** `workdir/model` (or set `MODEL_DIR`; HF: `<repo link>`)
- **Disk / VRAM:** `<X GiB / Y GiB>`
- **Backend:** vLLM (this is what the reported decode-throughput numbers were measured on)

## Run

```bash
bash setup.sh                                   # once: builds .venv, installs deps + modelzip
bash run.sh --lang-pair ces-deu --batch-size 8 --input in.txt --output out.txt
# positional form also works:  bash run.sh ces-deu 8 < in.txt > out.txt
```

Greedy decoding (`temperature=0`), stops at `<end_of_turn>`. One translation per input
line on stdout/`--output`; logs + progress on stderr.

## Env knobs (optional)

| var | default | effect |
|---|---|---|
| `MODELZIP_GPU_UTIL` | `0.90` | vLLM `gpu_memory_utilization` — lower keeps peak VRAM down |
| `MODELZIP_TP` | `1` | tensor-parallel size (honours `CUDA_VISIBLE_DEVICES`) |
| `MODELZIP_MAX_LEN` | `8192` | `max_model_len` — wmt25 inputs are paragraphs |

## Notes

This is a thin wrapper: `inference.py` subclasses `modelzip.submission.Gemma3LLMBase` and
overrides only `translate_batch` to use vLLM. All contract I/O, prompting, lang-pair
mapping, and batching come from the shared `modelzip` framework.
