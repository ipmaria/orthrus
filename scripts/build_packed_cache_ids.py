"""Pack on-policy samples given as token ids into the trainer's `input_ids`/`assistant_mask` cache.

Differences from scripts/build_packed_cache.py (text -> chat template -> tokens):

  * uses the exact tokens the base model generated (no re-tokenization drift);
  * truncated responses are kept as prefixes, without a fake <|im_end|>;
  * the WHOLE assistant span is supervised, <think>...</think> included;
  * rows start at a conversation boundary (bounded first-fit packing + padding),
    so the AR teacher never sees a response without its prompt.

    python scripts/build_packed_cache_ids.py \
        --dataset-path data/orthrus-onpolicy-think.jsonl \
        --out data/processed/orthrus-onpolicy-4096 \
        --model-dir pretrained-models/orthrus-qwen3-0.6b-init --seq-len 4096

Output loads with the unmodified trainer (`--packed-cache-path ... --seq-len 4096`).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys

import numpy as np

sys.path.insert(0, ".")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset-path", required=True,
                   help="JSONL from generate_onpolicy.py (prompt_ids/response_ids/finished).")
    p.add_argument("--out", required=True)
    p.add_argument("--model-dir", required=True, help="Orthrus init checkpoint (tokenizer).")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--open-rows", type=int, default=256,
                   help="Open rows kept for first-fit packing (bigger = tighter packing).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_samples(path, im_end, newline_ids, seq_len):
    samples, dropped = [], 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            prompt, resp = list(r["prompt_ids"]), list(r["response_ids"])
            if not resp or len(prompt) + 1 > seq_len - 1:
                dropped += 1
                continue
            ids = prompt + resp
            mask = [0] * len(prompt) + [1] * len(resp)
            if r.get("finished", True):
                ids += [im_end] + newline_ids          # supervise <|im_end|>, not the "\n"
                mask += [1] + [0] * len(newline_ids)
            if len(ids) > seq_len:                     # keep a prefix; it is still on-policy
                ids, mask = ids[:seq_len], mask[:seq_len]
            samples.append((ids, mask))
    return samples, dropped


def pack(samples, seq_len, pad_id, open_rows, rng):
    """Bounded first-fit: each sample goes to the first open row with room, else the
    fullest open row is closed. Rows never split a sample."""
    rng.shuffle(samples)
    rows, open_ = [], []
    for ids, mask in samples:
        for row in open_:
            if len(row[0]) + len(ids) <= seq_len:
                row[0].extend(ids)
                row[1].extend(mask)
                break
        else:
            if len(open_) >= open_rows:
                k = max(range(len(open_)), key=lambda i: len(open_[i][0]))
                rows.append(open_.pop(k))
            open_.append([list(ids), list(mask)])
    rows.extend(open_)
    used = sum(len(r[0]) for r in rows)
    for ids, mask in rows:
        pad = seq_len - len(ids)
        ids.extend([pad_id] * pad)
        mask.extend([0] * pad)
    return rows, used


def main():
    args = parse_args()
    import datasets
    from transformers import AutoTokenizer
    from src.utils.data_utils import load_packed_dataset

    if os.path.isdir(args.out):
        if not args.force:
            raise ValueError(f"{args.out} already exists; pass --force to overwrite.")
        shutil.rmtree(args.out)

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    newline_ids = tok.encode("\n", add_special_tokens=False)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    samples, dropped = load_samples(args.dataset_path, im_end, newline_ids, args.seq_len)
    print(f"samples={len(samples):,} dropped={dropped:,} (empty or prompt too long)")
    rows, used = pack(samples, args.seq_len, pad_id, args.open_rows, random.Random(args.seed))
    total = len(rows) * args.seq_len
    print(f"rows={len(rows):,} x {args.seq_len} | utilization={100 * used / total:.1f}% "
          f"(rest is padding, assistant_mask=0)")

    def gen():
        for ids, mask in rows:
            yield {"input_ids": np.asarray(ids, dtype=np.int32),
                   "assistant_mask": np.asarray(mask, dtype=np.int8)}

    features = datasets.Features({
        "input_ids": datasets.Sequence(datasets.Value("int32")),
        "assistant_mask": datasets.Sequence(datasets.Value("int8")),
    })
    datasets.Dataset.from_generator(gen, features=features).save_to_disk(args.out)

    ds = load_packed_dataset(args.out, args.seq_len)   # same checks the trainer runs
    head = ds.select(range(min(200, len(ds))))
    frac = sum(sum(r) for r in head["assistant_mask"]) / (len(head) * args.seq_len)
    print(f"assistant tokens ~{100 * frac:.1f}% of all tokens (first {len(head)} rows)")
    print(f"\nOK -- train with --packed-cache-path {args.out} --seq-len {args.seq_len}")


if __name__ == "__main__":
    main()
