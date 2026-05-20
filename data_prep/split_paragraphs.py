#!/usr/bin/env python3
"""
Split WMT25 gen-MT documents into paragraph-level JSONL records.

Reads wmt25-genmt.jsonl (local path or URL), filters by source/target language,
and emits one JSONL line per paragraph. Paragraphs are delimited by blank lines
(\\n\\n) in src_text and in each reference.

Output fields per line: doc_id, paragraph_id, src_text, refs.

Usage:
  python split_paragraphs.py \\
    --src-lang cs \\
    --tgt-lang de_DE \\
    --jsonl https://raw.githubusercontent.com/wmt-conference/wmt25-general-mt/refs/heads/main/data/wmt25-genmt.jsonl \\
    -o wmt25.cs-de_DE.paragraphs.jsonl
"""
import argparse
import json
import sys
import urllib.request
from typing import Any, Dict, Iterator, List, TextIO


DEFAULT_JSONL = (
    "https://raw.githubusercontent.com/wmt-conference/wmt25-general-mt/"
    "refs/heads/main/data/wmt25-genmt.jsonl"
)


def open_jsonl_stream(path: str) -> TextIO:
    if path.startswith("http://") or path.startswith("https://"):
        return urllib.request.urlopen(path)  # type: ignore[return-value]
    return open(path, encoding="utf-8")


def split_paragraphs(text: str) -> List[str]:
    return text.split("\n\n")


def split_refs(refs: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    return {name: split_paragraphs(entry["ref"]) for name, entry in refs.items()}


def paragraph_record(
    doc_id: str,
    paragraph_id: int,
    src_paragraph: str,
    refs: Dict[str, Dict[str, Any]],
    ref_paragraphs: Dict[str, List[str]],
) -> Dict[str, Any]:
    out_refs: Dict[str, Dict[str, Any]] = {}
    for name in refs:
        out_refs[name] = ref_paragraphs[name][paragraph_id - 1]
    return {
        "doc_id": doc_id,
        "paragraph_id": paragraph_id,
        "src_text": src_paragraph,
        "refs": out_refs,
    }


def validate_paragraph_counts(
    doc_id: str,
    src_paragraphs: List[str],
    ref_paragraphs: Dict[str, List[str]],
) -> None:
    src_count = len(src_paragraphs)
    mismatches = [
        (name, len(parts))
        for name, parts in ref_paragraphs.items()
        if len(parts) != src_count
    ]
    if mismatches:
        details = ", ".join(f"{name}={count}" for name, count in mismatches)
        raise ValueError(
            f"{doc_id}: paragraph count mismatch (src={src_count}, {details})"
        )


def iter_paragraph_records(
    obj: Dict[str, Any],
) -> Iterator[Dict[str, Any]]:
    doc_id = obj["doc_id"]
    src_paragraphs = split_paragraphs(obj["src_text"])
    ref_paragraphs = split_refs(obj["refs"])
    validate_paragraph_counts(doc_id, src_paragraphs, ref_paragraphs)

    for paragraph_id, src_paragraph in enumerate(src_paragraphs, start=1):
        yield paragraph_record(
            doc_id, paragraph_id, src_paragraph, obj["refs"], ref_paragraphs
        )


def process(
    jsonl_path: str,
    src_lang: str,
    tgt_lang: str,
    out: TextIO,
) -> tuple[int, int]:
    docs = 0
    paragraphs = 0
    with open_jsonl_stream(jsonl_path) as fh:
        for raw_line in fh:
            if isinstance(raw_line, bytes):
                raw_line = raw_line.decode("utf-8")
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("src_lang") != src_lang or obj.get("tgt_lang") != tgt_lang:
                continue
            docs += 1
            for record in iter_paragraph_records(obj):
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                paragraphs += 1
    return docs, paragraphs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-lang", required=True, help="Source language code (e.g. cs)")
    ap.add_argument("--tgt-lang", required=True, help="Target language code (e.g. de_DE)")
    ap.add_argument("--jsonl", default=DEFAULT_JSONL, help="Input JSONL path or URL")
    ap.add_argument(
        "-o",
        "--output",
        default="-",
        help="Output JSONL path (default: stdout)",
    )
    args = ap.parse_args()

    out_fh: TextIO
    close_out = False
    if args.output == "-":
        out_fh = sys.stdout
    else:
        out_fh = open(args.output, "w", encoding="utf-8")
        close_out = True

    try:
        docs, paragraphs = process(args.jsonl, args.src_lang, args.tgt_lang, out_fh)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        if close_out:
            out_fh.close()

    print(
        f"Wrote {paragraphs} paragraphs from {docs} documents "
        f"({args.src_lang} -> {args.tgt_lang})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
