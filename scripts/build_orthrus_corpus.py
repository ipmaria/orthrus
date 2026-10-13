"""Build the prompt corpus from Nemotron-Post-Training-Dataset-v2 (paper Appendix A).

The dataset's domains ARE its splits (`math`, `code`, `stem`, `chat`, `multilingual_*`);
there is no `train` split. Each domain split is read in full from the local cache,
prompts are de-duplicated (the same question often appears with a reasoning-on and a
reasoning-off response), Nemotron's reasoning-control system prompts ("/think",
"/no_think", "detailed thinking on|off") are dropped, and everything is written out.

    python scripts/build_orthrus_corpus.py --inspect
    python scripts/build_orthrus_corpus.py --out data/orthrus-sft.jsonl             # all prompts
    python scripts/build_orthrus_corpus.py --out data/orthrus-sft.jsonl --per-domain 50000 --balance

Only the prompt (system + first user turn) is written: on-policy generation regenerates
the response anyway, and Nemotron's responses make the `chat` split ~6 GB.
`--keep-responses` writes full conversations instead.
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


def clean_system(content: str) -> str | None:
    if CONTROL_RE.search(content):
        content = CONTROL_RE.sub("", content).strip()
    return content if content.strip() else None


def prompt_of(messages) -> list | None:
    """system (optional, control strings dropped, empty dropped) + first user turn."""
    if not isinstance(messages, list):
        return None
    out = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if not isinstance(content, str):
            return None
        if role == "system" and not out:
            content = clean_system(content)
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


def full_conversation(messages) -> list | None:
    if not isinstance(messages, list):
        return None
    cleaned = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if not role or not isinstance(content, str) or not content.strip():
            continue
        if role == "system":
            content = clean_system(content)
            if not content:
                continue
        cleaned.append({"role": role, "content": content})
    ok = len(cleaned) >= 2 and cleaned[0]["role"] in ("system", "user")
    return cleaned if ok else None


def prompt_key(messages: list) -> str:
    user = next(m["content"] for m in messages if m["role"] == "user")
    return hashlib.sha1(user.strip().encode("utf-8")).hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", type=str, default=DEFAULT_SOURCE)
    p.add_argument("--split-map", type=str, default=DEFAULT_SPLIT_MAP,
                   help='"domain=split[,split...] ..." e.g. "math=math code=code chat=chat,stem"')
    p.add_argument("--out", type=str, default=None, help="Output JSONL (messages rows).")
    p.add_argument("--per-domain", type=int, default=0,
                   help="Cap per domain (random subset after de-duplication). 0 = take all.")
    p.add_argument("--balance", action="store_true",
                   help="Truncate every domain to the smallest one (paper's 1:1:1).")
    p.add_argument("--keep-responses", action="store_true",
                   help="Write full conversations instead of prompts only.")
    p.add_argument("--inspect", action="store_true",
                   help="Print splits, columns and stats on a random sample, then exit.")
    p.add_argument("--inspect-rows", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not args.inspect and not args.out:
        raise ValueError("--out is required unless --inspect is passed.")
    return args


def main():
    args = parse_args()
    import datasets

    rng = random.Random(args.seed)
    split_map = parse_split_map(args.split_map)
    available = datasets.get_dataset_split_names(args.source)
    print(f"source={args.source} splits={available}")
    missing = [s for splits in split_map.values() for s in splits if s not in available]
    if missing:
        raise ValueError(f"Unknown splits {missing}; fix --split-map.")

    if args.inspect:
        for domain, splits in split_map.items():
            for split in splits:
                ds = datasets.load_dataset(args.source, split=split)
                idx = rng.sample(range(len(ds)), min(args.inspect_rows, len(ds)))
                sample = ds.select(idx)
                reasoning = Counter(str(r.get("reasoning")) for r in sample)
                systems = [r["messages"][0]["content"] or "" for r in sample
                           if r["messages"] and r["messages"][0]["role"] == "system"]
                empty_sys = sum(1 for c in systems if not c.strip())
                control = sum(1 for c in systems if CONTROL_RE.search(c))
                turns = Counter(sum(1 for m in r["messages"] if m["role"] == "user") for r in sample)
                example = next((prompt_of(r["messages"]) for r in sample if prompt_of(r["messages"])), None)
                print(f"\n[{domain}] split={split} rows={len(ds):,} columns={ds.column_names} "
                      f"(random sample of {len(sample)})")
                print(f"  reasoning={dict(reasoning)} | system prompts: {len(systems)} total, "
                      f"{empty_sys} empty (dropped), {control} with Nemotron control string "
                      f"(stripped) | user turns={dict(turns)}")
                print(f"  example prompt: {json.dumps(example, ensure_ascii=False)[:300]}")
        return

    buckets: dict[str, list] = {}
    for domain, splits in split_map.items():
        seen, rows, dupes, bad = set(), [], 0, 0
        for split in splits:
            ds = datasets.load_dataset(args.source, split=split).select_columns(["messages"])
            for row in ds:  # full pass over the split
                msgs = full_conversation(row["messages"]) if args.keep_responses \
                    else prompt_of(row["messages"])
                if msgs is None:
                    bad += 1
                    continue
                key = prompt_key(msgs)
                if key in seen:
                    dupes += 1
                    continue
                seen.add(key)
                rows.append(msgs)
        print(f"[{domain}] unique prompts={len(rows):,} duplicates={dupes:,} "
              f"malformed={bad:,} (splits={splits})")
        if args.per_domain and len(rows) > args.per_domain:
            rows = rng.sample(rows, args.per_domain)
        buckets[domain] = rows

    if args.balance:
        n = min(len(r) for r in buckets.values())
        buckets = {d: rng.sample(r, n) for d, r in buckets.items()}

    out_rows = [{"messages": m, "domain": d} for d, rows in buckets.items() for m in rows]
    rng.shuffle(out_rows)
    with open(args.out, "w") as handle:
        for r in out_rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    sizes = ", ".join(f"{d}={len(r):,}" for d, r in buckets.items())
    print(f"wrote {len(out_rows):,} prompts to {args.out} ({sizes})")
    print(f"\nNext: python scripts/generate_onpolicy.py --in {args.out} "
          f"--out data/orthrus-onpolicy-think.jsonl --model Qwen/Qwen3-0.6B")


if __name__ == "__main__":
    main()
