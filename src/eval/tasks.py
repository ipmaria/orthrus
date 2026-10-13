"""Benchmark definitions for the paper's evaluation suite (4.1).

Table 1 spans eight tasks; all eight are here.

  kind="math"        GSM8K, MATH-500, AIME24, AIME25   -- TPF + accuracy
  kind="code"        HumanEval, MBPP                   -- TPF + pass@1 (executed)
  kind="generation"  Pseudo2code, LiveCodeBench-v5     -- TPF only, see below

Every task supplies a prompt builder and a dataset id. Ids are overridable
(`--hf-path` / `--hf-config`) because community mirrors move around.

Scoring notes, stated plainly so the numbers are interpretable:
  * Math tasks use \\boxed{} extraction with a light normaliser, falling back to the
    last number in the completion. This is close to, but not identical with, the
    symbolic equivalence checkers used by some harnesses; it is deliberately simple
    and auditable rather than maximally generous.
  * Code tasks execute model output. That only runs under `--allow-code-execution`,
    in a subprocess with a wall-clock timeout.
  * `kind="generation"` tasks are measured for efficiency only -- TPF, wall-clock
    speedup, and the exact-match losslessness check -- and report no accuracy. That
    is exactly what the paper asks of them: Pseudo2code and LiveCodeBench-v5 appear
    in Table 1 (TPF / speedup) and in NEITHER accuracy table. Scoring them would mean
    two more execution backends the paper never uses -- a C++ toolchain for
    Pseudo2code, whose references are C++ checked against stdin/stdout cases, and
    LiveCodeBench's zlib+base64 private-test decoder plus its sandbox. The
    losslessness check still applies, and it is the claim these two rows exist to
    stress: long-horizon code generation is where conditional drift would show.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

MATH_INSTRUCTION = (
    "Solve the problem. Reason step by step, and put your final answer within "
    "\\boxed{}."
)


@dataclass
class Task:
    name: str
    hf_path: str
    hf_config: str | None
    split: str
    build_prompt: Callable[[dict], str]
    gold: Callable[[dict], str]
    kind: str  # "math" | "code" | "generation"
    max_new_tokens: int
    # Code tasks only:
    entry_point: Callable[[dict], str] | None = None
    test_program: Callable[[dict, str], str] | None = None
    # Tasks whose rows do not come from a plain `load_dataset(path, config, split)`.
    # Signature: (hf_path, hf_config, limit) -> sequence of row dicts.
    load_rows: Callable[[str, str | None, int | None], object] | None = None


# Answer extraction / normalisation (math)

def extract_boxed(text: str) -> str | None:
    """Content of the LAST \\boxed{...}, brace-balanced."""
    marker = "\\boxed"
    start = text.rfind(marker)
    if start == -1:
        return None
    cursor = start + len(marker)
    while cursor < len(text) and text[cursor] != "{":
        if text[cursor] not in " \t":
            return None
        cursor += 1
    if cursor >= len(text):
        return None
    depth = 0
    for index in range(cursor, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[cursor + 1 : index]
    return None


_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(text: str) -> str | None:
    matches = _NUMBER.findall(text)
    return matches[-1].replace(",", "") if matches else None


def normalise_math(answer: str) -> str:
    """Light, auditable normalisation -- not a symbolic equivalence check."""
    text = answer.strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    text = text.replace("dfrac", "frac").replace("tfrac", "frac")
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mbox\{([^}]*)\}", r"\1", text)
    text = text.replace("^{\\circ}", "").replace("^\\circ", "")
    text = text.replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    text = text.replace(" ", "").rstrip(".")
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    # 0.50 -> 0.5, 12.0 -> 12
    if re.fullmatch(r"-?\d+\.\d+", text):
        text = text.rstrip("0").rstrip(".")
    text = text.replace(",", "")
    return text


def score_math(completion: str, gold: str) -> bool:
    predicted = extract_boxed(completion)
    if predicted is None:
        predicted = last_number(completion)
    if predicted is None:
        return False
    gold_answer = extract_boxed(gold) or gold
    return normalise_math(predicted) == normalise_math(gold_answer)


# Code extraction

_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(completion: str) -> str:
    blocks = _CODE_FENCE.findall(completion)
    return blocks[-1] if blocks else completion


# Task registry

def _gsm8k_gold(row: dict) -> str:
    answer = row["answer"]
    return answer.split("####")[-1].strip() if "####" in answer else answer.strip()


def _humaneval_program(row: dict, completion: str) -> str:
    return "\n".join([
        extract_code(completion),
        row["test"],
        f"check({row['entry_point']})",
    ])


def _mbpp_program(row: dict, completion: str) -> str:
    return "\n".join([extract_code(completion)] + list(row["test_list"]))


# Pseudo2code (LongProc, Ye et al. 2025b). `input_prompt` is already a complete,
# self-contained instruction -- pseudocode plus the C++ preamble and the fencing
# convention -- so it is used verbatim rather than re-wrapped.

def _pseudo2code_prompt(row: dict) -> str:
    return row["input_prompt"]


# LiveCodeBench-v5 (Jain et al. 2024). The hub repo is a loading-script dataset,
# and `datasets` 4.0 removed script execution, so the shards are read directly.
# `release_v5` is cumulative v1..v5 (~880 problems); the bare `v5` window is
# test5.jsonl alone. Both are selectable via --hf-config.
LCB_REPO = "livecodebench/code_generation_lite"
_LCB_SHARDS = ["test.jsonl", "test2.jsonl", "test3.jsonl",
               "test4.jsonl", "test5.jsonl", "test6.jsonl"]
LCB_RELEASES = {
    f"release_v{n}": _LCB_SHARDS[:n] for n in range(1, 7)
}
LCB_RELEASES.update({f"v{n}": [_LCB_SHARDS[n - 1]] for n in range(1, 7)})

# The fields actually needed to build a prompt. Dropping the rest matters: the
# private_test_cases blobs dominate these shards (release_v5 is ~4.3 GB on the wire).
LCB_COLUMNS = [
    "question_title", "question_content", "starter_code",
    "platform", "question_id", "difficulty",
]


def _lcb_prompt(row: dict) -> str:
    """LiveCodeBench's own two prompt forms: functional vs stdin/stdout."""
    starter = (row.get("starter_code") or "").strip()
    parts = [
        "You will be given a question (problem specification) and will generate a "
        "correct Python program that matches the specification and passes all tests.",
        f"### Question:\n{row['question_content']}",
    ]
    if starter:
        parts.append(
            "### Format: You will use the following starter code to write the "
            "solution to the problem and enclose your code within delimiters.\n"
            f"```python\n{starter}\n```"
        )
    else:
        parts.append(
            "### Format: Read the inputs from stdin, solve the problem and write the "
            "answer to stdout (do not directly test on the sample inputs). Enclose "
            "your code within delimiters as follows.\n"
            "```python\n# YOUR CODE HERE\n```"
        )
    parts.append("### Answer: (use the provided format with backticks)")
    return "\n\n".join(parts)


def _lcb_load_rows(hf_path: str, hf_config: str | None, limit: int | None):
    """Stream the release's jsonl shards straight from the hub.

    Streaming is not just an optimisation here: with --limit it stops after the
    requested rows instead of pulling every shard in full.
    """
    release = hf_config or "release_v5"
    if release not in LCB_RELEASES:
        raise ValueError(
            f"Unknown LiveCodeBench release {release!r}. "
            f"Choose one of: {', '.join(sorted(LCB_RELEASES))}."
        )

    import datasets

    data_files = [
        f"hf://datasets/{hf_path}/{shard}" for shard in LCB_RELEASES[release]
    ]
    stream = datasets.load_dataset(
        "json", data_files=data_files, split="train", streaming=True
    )
    if limit is not None:
        stream = stream.take(limit)
    # Project in Python rather than `select_columns`: on a streaming JSON dataset the
    # features are inferred lazily, and this keeps the blobs out of the row list either
    # way. `question_content` is the only field the prompt cannot do without.
    rows = [{key: row.get(key) for key in LCB_COLUMNS} for row in stream]
    if not rows:
        raise RuntimeError(f"LiveCodeBench release {release} came back empty.")
    return rows


TASKS: dict[str, Task] = {
    "gsm8k": Task(
        name="gsm8k",
        hf_path="openai/gsm8k", hf_config="main", split="test",
        build_prompt=lambda row: f"{MATH_INSTRUCTION}\n\n{row['question']}",
        gold=_gsm8k_gold, kind="math", max_new_tokens=4096,
    ),
    "math500": Task(
        name="math500",
        hf_path="HuggingFaceH4/MATH-500", hf_config=None, split="test",
        build_prompt=lambda row: f"{MATH_INSTRUCTION}\n\n{row['problem']}",
        gold=lambda row: str(row["answer"]), kind="math", max_new_tokens=8192,
    ),
    "aime24": Task(
        name="aime24",
        hf_path="HuggingFaceH4/aime_2024", hf_config=None, split="train",
        build_prompt=lambda row: f"{MATH_INSTRUCTION}\n\n{row['problem']}",
        gold=lambda row: str(row["answer"]), kind="math", max_new_tokens=16384,
    ),
    "aime25": Task(
        name="aime25",
        hf_path="yentinglin/aime_2025", hf_config=None, split="train",
        build_prompt=lambda row: f"{MATH_INSTRUCTION}\n\n{row['problem']}",
        gold=lambda row: str(row["answer"]), kind="math", max_new_tokens=16384,
    ),
    "humaneval": Task(
        name="humaneval",
        hf_path="openai/openai_humaneval", hf_config=None, split="test",
        build_prompt=lambda row: (
            "Complete the following Python function. Return the full function in a "
            f"```python code block.\n\n```python\n{row['prompt']}```"
        ),
        gold=lambda row: row["entry_point"], kind="code", max_new_tokens=4096,
        entry_point=lambda row: row["entry_point"],
        test_program=_humaneval_program,
    ),
    "mbpp": Task(
        name="mbpp",
        hf_path="google-research-datasets/mbpp", hf_config="sanitized", split="test",
        build_prompt=lambda row: (
            f"{row['prompt' if 'prompt' in row else 'text']}\n\nYour solution must "
            f"satisfy these tests:\n" + "\n".join(row["test_list"]) +
            "\n\nReturn the complete solution in a ```python code block."
        ),
        gold=lambda row: "\n".join(row["test_list"]), kind="code", max_new_tokens=4096,
        test_program=_mbpp_program,
    ),
    # Table 1's last two rows. Efficiency-only -- see the module docstring.
    "pseudo2code": Task(
        name="pseudo2code",
        # LongProc ships 0.5k and 2k output-length variants; the paper does not say
        # which. 2k is the default; `--hf-config pseudo_to_code_0.5k` selects the other.
        hf_path="PrincetonPLI/LongProc", hf_config="pseudo_to_code_2k", split="test",
        build_prompt=_pseudo2code_prompt,
        gold=lambda row: row["reference_output"], kind="generation", max_new_tokens=8192,
    ),
    "livecodebench_v5": Task(
        name="livecodebench_v5",
        hf_path=LCB_REPO, hf_config="release_v5", split="test",
        build_prompt=_lcb_prompt,
        gold=lambda row: row["question_id"], kind="generation", max_new_tokens=8192,
        load_rows=_lcb_load_rows,
    ),
}


def load_task_rows(
    task: Task,
    limit: int | None = None,
    hf_path: str | None = None,
    hf_config: str | None = None,
):
    import datasets

    path = hf_path or task.hf_path
    config = hf_config or task.hf_config

    if task.load_rows is not None:
        return task.load_rows(path, config, limit)

    try:
        rows = datasets.load_dataset(path, config, split=task.split)
    except Exception as error:  # dataset moved / renamed / gated
        raise RuntimeError(
            f"Could not load `{path}` (config={config}, split={task.split}) for "
            f"task {task.name}: {error}\nPass --hf-path to point at a mirror."
        ) from error
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    return rows


def run_code_sample(program: str, timeout: float = 10.0) -> bool:
    """Execute one candidate program in a subprocess. Returns True iff it exits 0."""
    import subprocess
    import sys
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(program)
        path = handle.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, timeout=timeout, text=True,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        import os

        try:
            os.unlink(path)
        except OSError:
            pass
