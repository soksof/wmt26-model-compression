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
| `MODELZIP_MBR` | `0` | `N>1` → self-MBR decode (chrF consensus of `N` samples). `0`/`1` = greedy, PRIMARY byte-identical |
| `MODELZIP_MBR_MINP` | `0.02` | min_p tail-prune for MBR sampling (vLLM's epsilon-sampling stand-in) |
| `MODELZIP_MBR_TEMP` | `1.0` | MBR sampling temperature |
| `MODELZIP_MBR_SEED` | `0` | MBR sampling RNG seed (reproducible candidate sets) |

## MBR quality contrastive (en→ar only)

The self-MBR point is **a decode flag on the en→ar PRIMARY checkpoint — zero new
checkpoints, same disk/VRAM** (chrF-consensus of N samples; §5.5). It ships **en→ar only**
(it helps only the weakest direction) and costs ~`N`× decode, so its throughput is **not** a
speed-axis number — mark it unambiguously as a *contrastive* so its decode cost never attaches
to a PRIMARY's frontier position.

**Assemble the contrastive package** — a normal per-system copy of this dir, with the flag baked
into the artifact (via `submission.env`, which `run.sh` sources) so the *organizers'* run
reproduces it, not just yours:

```bash
cp -r submit/vllm submissions/en-ar-mbr                         # the package
cp submissions/en-ar-mbr/submission.env.example \
   submissions/en-ar-mbr/submission.env                         # bakes MODELZIP_MBR=16
ln -s "$(realpath models/gemma-3-12b-vocaball-int4-cal196)" \
   submissions/en-ar-mbr/workdir/model                          # the en-ar PRIMARY checkpoint
cd submissions/en-ar-mbr && bash setup.sh                       # venv + modelzip framework
```

Then validate/run for `eng-ara_EG` like any package (`modelzip.setup` / `modelzip.evaluate`, or
`bash run.sh --lang-pair eng-ara_EG ...`). The sampling config (N=16, min_p=0.02, temp=1.0,
**seed=0**) is fixed in `inference.py`'s defaults, so the candidate set is reproducible from the
artifact alone. Greedy (no `submission.env`) leaves the PRIMARY package provably untouched — PRIMARY
and contrastive are the same code + checkpoint, differing only by that one file.

**Close the loop (§6):** after the harness pass, assert COMET on the modelzip-produced output ≈ your
Gate 1 MBR run (within the noise band) — seeded sampling is reproducible but not bit-identical across
kernels, so verify rather than assume.

For a **quick local check** without a separate package, the env var also works directly:
```bash
MODELZIP_MBR=16 bash run.sh --lang-pair eng-ara_EG --batch-size 8 --input in.txt --output out.txt
```

## Notes

This is a thin wrapper: `inference.py` subclasses `modelzip.submission.Gemma3LLMBase` and
overrides only `translate_batch` to use vLLM. All contract I/O, prompting, lang-pair
mapping, and batching come from the shared `modelzip` framework.
