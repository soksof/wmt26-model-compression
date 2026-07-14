#!/usr/bin/env python3
"""WMT26 model-compression submission — vLLM backend for our compressed Gemma 3
checkpoints (FP8 / INT4-W4A16 / depth-pruned / vocab-pruned).

Conforms to the modelzip contract by plugging into ``modelzip.submission.run_inference``,
which already handles everything the contract requires: ``--lang-pair`` / ``--batch-size``
/ ``--input`` / ``--output``, source-lines-in -> one-translation-per-line-out, the
``TRANSLATE_PROMPT`` + chat template, lang-pair mapping, length-sorted batching, and the
per-batch token budget. We override ONLY the generation backend (transformers -> vLLM) so
the decode throughput matches the numbers our frontier is measured on.

The same file serves every direction the checkpoint supports — direction is the runtime
``--lang-pair`` arg, so one submission dir per *model*; point ``MODEL_DIR`` at the checkpoint.

Optional env knobs:
  MODELZIP_TP        tensor-parallel size (default 1; vLLM honours CUDA_VISIBLE_DEVICES)
  MODELZIP_GPU_UTIL  vLLM gpu_memory_utilization (default 0.90; lower -> lower peak VRAM)
  MODELZIP_MAX_LEN   vLLM max_model_len (default 8192; wmt25 inputs are paragraphs)
  MODELZIP_SENTENCE_SPLIT  =1 -> split each source paragraph into sentences, translate each,
                     rejoin to one output line. For depth-pruned students that lose fidelity
                     on long paragraphs but translate sentences faithfully. (needs pysbd for
                     the best split; falls back to a punctuation regex.)
  MODELZIP_MBR       N>1 -> Minimum Bayes Risk decode: sample N candidates from THIS model and
                     ship the one with highest mean chrF agreement (a decode-only QUALITY
                     contrastive; same model/disk/VRAM as the greedy PRIMARY). Unset / 0 / 1 ->
                     greedy, so the PRIMARY is provably untouched. Needs sacrebleu.
  MODELZIP_MBR_MINP  min_p tail-prune for MBR sampling (default 0.02; vLLM's native stand-in for
                     epsilon sampling -- there is no epsilon_cutoff knob in vLLM's SamplingParams).
  MODELZIP_MBR_TEMP  sampling temperature for MBR candidates (default 1.0).
  MODELZIP_MBR_SEED  RNG seed for reproducible MBR candidate sets (default 0).
"""
from __future__ import annotations

import os
import re

from modelzip.submission import (
    DEF_MAX_NEW_TOKENS,
    DEF_MAX_NEW_TOKENS_OVER_INPUT,
    Gemma3LLMBase,
    TRANSLATE_PROMPT,
    default_model_path,
    parse_inference_args,
    run_inference,
)


_SENT_FALLBACK = re.compile(r"(?<=[.!?…])\s+")
# source-side lang for the splitter (we only ever split the SOURCE, which is always Latin:
# ces/eng). Targets (deu/zho/ara) are never split.
_PYSBD_LANG = {"ces": "cs", "eng": "en", "deu": "de"}


def _split_sentences(text: str, lang: str) -> list[str]:
    """Split a source paragraph into sentences for the optional MODELZIP_SENTENCE_SPLIT path.

    Depth-pruned students translate sentences faithfully but lose fidelity on long
    paragraphs (fewer layers -> less long-context capacity); feeding them one sentence at a
    time recovers quality. Uses pysbd when present (handles abbreviations like 'z. B.'),
    else a punctuation-regex fallback. Never returns []: a non-splittable line -> [line]."""
    text = text.strip()
    if not text:
        return []
    for L in (lang, "en"):
        try:
            import pysbd

            segs = [s.strip() for s in pysbd.Segmenter(language=L, clean=False).segment(text) if s.strip()]
            if segs:
                return segs
        except Exception:  # noqa: BLE001  pysbd missing or lang unsupported -> try next / regex
            continue
    return [p.strip() for p in _SENT_FALLBACK.split(text) if p.strip()] or [text]


class VllmGemma3LLM(Gemma3LLMBase):
    """Inherits the contract machinery (make_prompt, source_length, translate_lines,
    lang-pair mapping) from the framework; swaps the generation backend to vLLM.

    The inherited transformers ``model`` property is never touched — we override
    ``translate_batch`` and drive a lazily-built vLLM engine instead."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vllm = None
        self._eot = None  # <end_of_turn> stop id, resolved at engine init

    @property
    def vllm(self):
        if self._vllm is None:
            from vllm import LLM

            engine = LLM(
                model=str(self.model_dir),
                dtype="auto",  # bf16 / compressed-tensors (INT4/FP8) auto-detected from config
                tensor_parallel_size=int(os.getenv("MODELZIP_TP", "1")),
                gpu_memory_utilization=float(os.getenv("MODELZIP_GPU_UTIL", "0.90")),
                max_model_len=int(os.getenv("MODELZIP_MAX_LEN", "8192")),
                # MODELZIP_ENFORCE_EAGER=1 skips torch.compile -> fast startup (no slow
                # first-run inductor compile), slightly slower decode. Use for quick
                # validation; leave 0 for the submission so the throughput run is compiled.
                enforce_eager=os.getenv("MODELZIP_ENFORCE_EAGER", "0") == "1",
            )
            # Gemma emits <end_of_turn> (id 106 in the full vocab) to end its turn; vLLM
            # stops on <eos>=1 by default, NOT on 106 -> resolve it from the tokenizer
            # (robust to a vocab-pruned tokenizer that could remap it) and stop on it too.
            eot = engine.get_tokenizer().convert_tokens_to_ids("<end_of_turn>")
            self._eot = [eot] if isinstance(eot, int) and eot >= 0 else None
            self._vllm = engine
        return self._vllm

    @property
    def length_tokenizer(self):
        # modelzip's source_length() calls self.length_tokenizer(text, ...) just to
        # length-sort/budget batches. Route it through vLLM's already-loaded tokenizer
        # rather than a fresh AutoTokenizer: that avoids (a) the multimodal-processor
        # path on our text-only models and (b) the transformers-5.x Gemma load bug
        # (`extra_special_tokens` list vs dict). vLLM loads this tokenizer fine.
        return self.vllm.get_tokenizer()

    def _generate(self, pair, texts, max_new_tokens):
        """Greedy-translate each text -> one line (newlines collapsed). The raw backend
        call, shared by the default path and the sentence-split path."""
        from vllm import SamplingParams

        engine = self.vllm  # ensure engine + stop-id are initialised
        prompts = [self.make_prompt(pair, text) for text in texts]
        conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
        params = SamplingParams(
            temperature=0.0,  # greedy, matches the baseline's do_sample=False/num_beams=1
            max_tokens=max_new_tokens,
            stop_token_ids=self._eot,
        )
        outputs = engine.chat(conversations, params, use_tqdm=self.progress_bar)
        # vLLM returns RequestOutputs in input order; collapse internal newlines so each
        # source line maps to exactly one output line (matches Gemma3LLMBase.generate_gemma3).
        return [out.outputs[0].text.replace("\n", " ").strip() for out in outputs]

    def _generate_mbr(self, pair, texts, max_new_tokens, n):
        """Minimum Bayes Risk decode each text -> one line. Sample ``n`` candidates from THIS
        model and return the consensus one -- the candidate maximising mean chrF against the
        others (Eikema & Aziz self-/consensus MBR, chrF utility). No external model or data:
        pure decode-time algorithm on the same checkpoint, so it's a QUALITY contrastive that
        differs from the greedy PRIMARY only by run.sh flags -- zero new checkpoints/uploads.

        Cost is ~n*decode, NOT n*(prefill+decode): vLLM's ``n=`` forks n sampled sequences
        after a SINGLE shared prefill of the (identical-per-input) prompt, so on this
        prefill-bound workload the wall-time multiplier is well under n."""
        from vllm import SamplingParams
        from sacrebleu.metrics import CHRF

        engine = self.vllm  # ensure engine + stop-id are initialised
        temp = float(os.getenv("MODELZIP_MBR_TEMP", "1.0"))
        min_p = float(os.getenv("MODELZIP_MBR_MINP", "0.02"))
        seed = int(os.getenv("MODELZIP_MBR_SEED", "0"))
        prompts = [self.make_prompt(pair, text) for text in texts]
        conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
        # min_p is vLLM 0.23's native tail-prune (there is no epsilon_cutoff): it drops the
        # low-probability tail that poisons pairwise utilities at paragraph length, which is
        # exactly why epsilon sampling beats nucleus/temperature for MBR (Freitag et al.).
        params = SamplingParams(
            n=n,
            temperature=temp,
            min_p=min_p,
            seed=seed,
            max_tokens=max_new_tokens,
            stop_token_ids=self._eot,
        )
        outputs = engine.chat(conversations, params, use_tqdm=self.progress_bar)
        chrf = CHRF()  # sentence-level char-n-gram F; reused across all comparisons
        results = []
        for out in outputs:
            cands = [o.text.replace("\n", " ").strip() for o in out.outputs]
            results.append(self._mbr_select(cands, chrf))
        return results

    @staticmethod
    def _mbr_select(cands, chrf):
        """Pick the MBR consensus candidate: argmax_h mean_r chrF(h, r).

        Empties are dropped (a derailed/empty sample must neither win nor skew the utility).
        Hypotheses are deduped (no point scoring an identical string twice), but the pseudo-
        reference set KEEPS duplicates so a more-probable candidate weights the expectation
        correctly -- the samples are the empirical model distribution MBR integrates over."""
        refs = [c for c in cands if c]
        if not refs:
            return ""
        uniq = list(dict.fromkeys(refs))  # dedup hypotheses, preserve first-seen order
        if len(uniq) == 1:
            return uniq[0]
        best, best_score = uniq[0], -1.0
        for h in uniq:
            score = sum(chrf.sentence_score(h, [r]).score for r in refs) / len(refs)
            if score > best_score:
                best, best_score = h, score
        return best

    def translate_batch(self, pair, texts, max_new_tokens):
        # MODELZIP_MBR=N>1: quality-contrastive MBR path (opt-in). Takes precedence over the
        # sentence-split path and leaves the default greedy path below byte-identical when unset.
        mbr_n = int(os.getenv("MODELZIP_MBR", "0") or "0")
        if mbr_n > 1:
            return self._generate_mbr(pair, texts, max_new_tokens, mbr_n)
        # Default: one generation per input line (paragraph) -> one output line.
        if os.getenv("MODELZIP_SENTENCE_SPLIT") != "1":
            return self._generate(pair, texts, max_new_tokens)
        # MODELZIP_SENTENCE_SPLIT=1: split each paragraph into sentences, translate them all
        # (flattened into one batch so vLLM still batches efficiently), then rejoin per input
        # line with spaces -> still exactly one output line per input line (contract-safe).
        # Salvages depth-pruned students that degrade on long paragraphs.
        src = _PYSBD_LANG.get(pair.split("-")[0], "en")
        flat, owner = [], []
        for i, text in enumerate(texts):
            for sent in _split_sentences(text, src) or [text]:
                flat.append(sent)
                owner.append(i)
        pieces = self._generate(pair, flat, max_new_tokens) if flat else []
        joined: list[list[str]] = [[] for _ in texts]
        for owner_idx, piece in zip(owner, pieces):
            joined[owner_idx].append(piece)
        return [" ".join(parts).strip() for parts in joined]


def parse_args():
    return parse_inference_args(
        default_model=default_model_path(__file__),
        description="WMT26 compression — vLLM Gemma 3 (FP8 / INT4-W4A16 / depth / vocab)",
        default_prompt=TRANSLATE_PROMPT,
        default_max_new_tokens=DEF_MAX_NEW_TOKENS,
        default_max_new_tokens_over_input=DEF_MAX_NEW_TOKENS_OVER_INPUT,
    )


def main():
    run_inference(parse_args(), VllmGemma3LLM, use_chat_template=True)


if __name__ == "__main__":
    main()
