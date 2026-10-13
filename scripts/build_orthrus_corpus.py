"""Build the paper's training corpus (Appendix A): domain-balanced `messages` rows.

    # inspect the source schema first
    python scripts/build_orthrus_corpus.py --inspect

    # then sample 1:1:1 chat / math / code and write a messages JSONL
    python scripts/build_orthrus_corpus.py --out data/orthrus-sft.jsonl --per-domain 200000

Appendix A: "sampling from the open-source Nemotron-Post-Training-Dataset-v2 ...
strictly balanced across three domains: Mathematical Reasoning, Code Generation, and
General Chat & Instruction Tuning", packed to L=2048 for 471,952 rows / 0.96B tokens.

The upstream dataset's split and category naming has changed over time, so the domain
mapping is data-driven (`--inspect` prints what is actually there) rather than
hard-coded. Feed the output to `scripts/build_packed_cache.py` to get the packed cache
the trainer consumes.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

DEFAULT_SOURCE = "nvidia/Nemotron-Post-Training-Dataset-v2"

# Substrings matched (lowercased) against the dataset's category/split label.
DOMAIN_PATTERNS = {
    "math": ("math", "aime", "gsm", "reasoning_math"),
    "code": ("code", "coding", "program", "swe"),
    "chat": ("chat", "instruct", "general", "if", "stem", "science"),
}


def classify(label: str) -> str | None:
    lowered = str(label).lower()
    for domain, patterns in DOMAIN_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            return domain
    return None


def find_label_column(columns) -> str | None:
    for candidate in ("category", "domain", "task", "source", "subset", "dataset"):
        if candidate in columns:
            return candidate
    return None


def to_messages(row, label_column) -> list | None:
    """Normalise a row to a `messages` list of {role, content}."""
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        cleaned = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role and isinstance(content, str) and content.strip():
                cleaned.append({"role": role, "content": content})
        return cleaned if len(cleaned) >= 2 else None

    # Fall back to prompt/response style columns.
    for prompt_key, response_key in (("input", "output"), ("prompt", "response"),
                                     ("question", "answer"), ("instruction", "output")):
        if prompt_key in row and response_key in row:
            prompt, response = row[prompt_key], row[response_key]
            if isinstance(prompt, str) and isinstance(response, str) and prompt and response:
                return [{"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}]
    return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", type=str, default=DEFAULT_SOURCE)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--out", type=str, default=None, help="Output JSONL (messages rows).")
    p.add_argument("--per-domain", type=int, default=200_000,
                   help="Conversations per domain; 1:1:1 across math/code/chat.")
    p.add_argument("--inspect", action="store_true",
                   help="Print columns and category counts, then exit.")
    p.add_argument("--inspect-rows", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--streaming", action="store_true",
                   help="Stream the source instead of downloading it whole.")
    args = p.parse_args()
    if not args.inspect and not args.out:
        raise ValueError("--out is required unless --inspect is passed.")
    return args


def main():
    args = parse_args()
    import datasets

    dataset = datasets.load_dataset(args.source, split=args.split, streaming=args.streaming)

    if args.inspect:
        head = list(dataset.take(args.inspect_rows)) if args.streaming else \
            list(dataset.select(range(min(args.inspect_rows, len(dataset)))))
        columns = sorted({key for row in head for key in row})
        label_column = find_label_column(columns)
        print(f"source={args.source} split={args.split}")
        print(f"columns={columns}")
        print(f"label column -> {label_column}")
        if label_column:
            counts = Counter(str(row.get(label_column)) for row in head)
            print(f"\ncategory counts (first {len(head)} rows):")
            for label, count in counts.most_common(40):
                print(f"  {label:<45} {count:>7}  -> {classify(label) or 'UNMAPPED'}")
        return

    columns = dataset.column_names or []
    label_column = find_label_column(columns)
    if label_column is None:
        raise ValueError(
            f"No category-like column in {columns}. Run --inspect and extend "
            f"DOMAIN_PATTERNS / find_label_column for this source."
        )

    random.seed(args.seed)
    buckets: dict[str, list] = defaultdict(list)
    skipped = 0

    for row in dataset:
        domain = classify(row.get(label_column, ""))
        if domain is None or len(buckets[domain]) >= args.per_domain:
            if domain is None:
                skipped += 1
            if all(len(buckets[name]) >= args.per_domain for name in DOMAIN_PATTERNS):
                break
            continue
        messages = to_messages(row, label_column)
        if messages is None:
            skipped += 1
            continue
        buckets[domain].append(messages)

    per_domain = min(len(buckets[name]) for name in DOMAIN_PATTERNS)
    if per_domain == 0:
        raise ValueError(f"One domain came back empty: "
                         f"{{k: len(v) for k, v in buckets.items()}}. Run --inspect.")
    if per_domain < args.per_domain:
        print(f"NOTE: only {per_domain:,} conversations available in the smallest domain; "
              f"truncating all three to that for a strict 1:1:1 balance.")

    rows = []
    for domain in DOMAIN_PATTERNS:
        rows.extend(buckets[domain][:per_domain])
    random.shuffle(rows)

    with open(args.out, "w") as handle:
        for messages in rows:
            handle.write(json.dumps({"messages": messages}) + "\n")

    print(f"wrote {len(rows):,} conversations to {args.out} "
          f"({per_domain:,} per domain x 3; skipped {skipped:,} unmapped/malformed)")
    print(f"\nNext: python scripts/build_packed_cache.py --dataset-path {args.out} \\\n"
          f"        --out data/processed/orthrus-packed-2048-asst \\\n"
          f"        --model-dir pretrained-models/orthrus-qwen3-8b-init --seq-len 2048")


if __name__ == "__main__":
    main()
