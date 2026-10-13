"""Build the prompt corpus (paper Appendix A): domain-balanced `messages` rows from
Nemotron-Post-Training-Dataset-v2.

The dataset's domains ARE its splits (`math`, `code`, `stem`, `chat`, `multilingual_*`);
there is no `train` split and no category column to classify. Each domain is loaded as
its own split, prompts are de-duplicated, Nemotron's reasoning-control system prompts
("/think", "/no_think", "detailed thinking on|off") are dropped, and the three domains
are truncated to the same size (1:1:1, as in the paper).

    python scripts/build_orthrus_corpus.py --inspect
    python scripts/build_orthrus_corpus.py --out data/orthrus-sft.jsonl --per-domain 200000

Only the prompt (system + first user turn) is written by default: on-policy generation
(scripts/generate_onpolicy.py) regenerates the response anyway, and Nemotron's responses
make the `chat` split ~6 GB. `--keep-responses` writes full conversations instead.

The `code` split has 175,000 rows, so the paper's 200K per domain is not reachable for
code; the 1:1:1 balance then caps every domain at what survives de-duplication there.
`--split-map chat=chat,stem` folds the STEM split into the chat bucket if you want more.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter

DEFAULT_SOURCE = "nvidia/Nemotron-Post-Training-Dataset-v2"
DEFAULT_SPLIT_MAP = "math=math code=code chat=chat"

# Nemotron-specific reasoning switches; meaningless for Qwen3 (which has its own soft
# switch that these strings would trigger) and absent from any prompt at eval time.
CONTROL_RE = re.compile(r"(/think\b|/no_think\b|detailed thinking (on|off))", re.IGNORECASE)


def parse_split_map(spec: str) -> dict[str, list[str]]:
    out = {}
    for item in spec.split():
        domain, splits = item.split("=", 1)
        out[domain] = [s for s in splits.split(",") if s]
    return out


def prompt_of(row) -> list | None:
    """system (optional, control strings dropped) + first user turn."""
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    out = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if not isinstance(content, str):
            return None
        if role == "system" and not out:
            if CONTROL_RE.search(content):
                content = CONTROL_RE.sub("", content).strip()
            if content:
                out.append({"role": "system", "content": content})
        elif role == "user":
            if not content.strip():
                return None
            out.append({"role": "user", "content": content})
            return out
        else:
            return None
    return None


def full_conversation(row) -> list | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    cleaned = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if role and isinstance(content, str) and content.strip():
            if role == "system" and CONTROL_RE.search(content):
                content = CONTROL_RE.sub("", content).strip()
                if not content:
                    continue
            cleaned.append({"role": role, "content": content})
    return cleaned if len(cleaned) >= 2 and cleaned[0]["role"] in ("system", "user") else None


def prompt_key(messages: list) -> str:
    user = next(m["content"] for m in messages if m["role"] == "user")
    return hashlib.sha1(user.strip().encode("utf-8")).hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", type=str, default=DEFAULT_SOURCE)
    p.add_argument("--split-map", type=str, default=DEFAULT_SPLIT_MAP,
                   help='"domain=split[,split...] ..." e.g. "math=math code=code chat=chat,stem"')
    p.add_argument("--out", type=str, default=None, help="Output JSONL (messages rows).")
    p.add_argument("--per-domain", type=int, default=200_000,
                   help="Conversations per domain; the smallest domain caps the others (1:1:1).")
    p.add_argument("--no-balance", action="store_true",
                   help="Keep every domain at min(--per-domain, available) instead of 1:1:1.")
    p.add_argument("--keep-responses", action="store_true",
                   help="Write full conversations instead of prompts only.")
    p.add_argument("--inspect", action="store_true",
                   help="Print splits, columns and a few stats, then exit.")
    p.add_argument("--inspect-rows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--streaming", action="store_true",
                   help="Stream the parquet shards instead of materialising the splits "
                        "(no multi-GB download; recommended on a first run).")
    args = p.parse_args()
    if not args.inspect and not args.out:
        raise ValueError("--out is required unless --inspect is passed.")
    return args


def load_split(source: str, split: str, streaming: bool):
    import datasets
    return datasets.load_dataset(source, split=split, streaming=streaming)


def main():
    args = parse_args()
    import datasets

    split_map = parse_split_map(args.split_map)
    available = datasets.get_dataset_split_names(args.source)
    print(f"source={args.source} splits={available}")
    missing = [s for splits in split_map.values() for s in splits if s not in available]
    if missing:
        raise ValueError(f"Unknown splits {missing}; fix --split-map.")

    if args.inspect:
        for domain, splits in split_map.items():
            for split in splits:
                ds = load_split(args.source, split, streaming=True)
                head = list(ds.take(args.inspect_rows))
                columns = sorted({k for row in head for k in row})
                reasoning = Counter(str(r.get("reasoning")) for r in head)
                with_system = sum(1 for r in head if r["messages"] and r["messages"][0]["role"] == "system")
                control = sum(1 for r in head if r["messages"] and r["messages"][0]["role"] == "system"
                              and CONTROL_RE.search(r["messages"][0]["content"] or ""))
                turns = Counter(sum(1 for m in r["messages"] if m["role"] == "user") for r in head)
                print(f"\n[{domain}] split={split} columns={columns}")
                print(f"  reasoning={dict(reasoning)} | rows with system prompt={with_system} "
                      f"(with Nemotron control string: {control}) | user turns={dict(turns)}")
                example = next((prompt_of(r) for r in head if prompt_of(r)), None)
                print(f"  example prompt: {json.dumps(example, ensure_ascii=False)[:300]}")
        return

    rng = random.Random(args.seed)
    buckets: dict[str, list] = {}
    for domain, splits in split_map.items():
        seen, rows, dupes, bad = set(), [], 0, 0
        for split in splits:
            ds = load_split(args.source, split, streaming=args.streaming)
            for row in ds:
                msgs = full_conversation(row) if args.keep_responses else prompt_of(row)
                if msgs is None:
                    bad += 1
                    continue
                key = prompt_key(msgs)
                if key in seen:
                    dupes += 1
                    continue
                seen.add(key)
                rows.append(msgs)
                if len(rows) >= args.per_domain:
                    break
            if len(rows) >= args.per_domain:
                break
        rng.shuffle(rows)
        buckets[domain] = rows
        print(f"[{domain}] kept={len(rows):,} duplicates={dupes:,} malformed={bad:,} "
              f"(splits={splits})")

    if args.no_balance:
        sizes = {d: len(r) for d, r in buckets.items()}
    else:
        n = min(len(r) for r in buckets.values())
        sizes = {d: n for d in buckets}
        if n < args.per_domain:
            print(f"NOTE: smallest domain has {n:,}; all domains truncated to that for 1:1:1.")

    out_rows = []
    for domain, rows in buckets.items():
        out_rows.extend({"messages": m, "domain": domain} for m in rows[:sizes[domain]])
    rng.shuffle(out_rows)
    with open(args.out, "w") as handle:
        for r in out_rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out_rows):,} prompts to {args.out} "
          f"({', '.join(f'{d}={n:,}' for d, n in sizes.items())})")
    print(f"\nNext: python scripts/generate_onpolicy.py --in {args.out} "
          f"--out data/orthrus-onpolicy-think.jsonl --model Qwen/Qwen3-0.6B")


if __name__ == "__main__":
    main()
