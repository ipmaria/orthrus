"""Tree verification of several parallel drafts (src/tree_verify.py). Torch-only."""

from __future__ import annotations

from itertools import product

import torch

from src.tree_verify import tree_speculative_walk, tree_verification_mask


def test_tree_mask_sees_prefix_and_own_chain_causally():
    mask = tree_verification_mask(num_chains=2, block_len=3, prefix_len=4,
                                  dtype=torch.float32)
    allowed = mask[0, 0] == 0
    assert allowed.shape == (6, 10)
    assert allowed[:, :4].all()                         # the shared committed prefix
    own = allowed[:, 4:]
    expected = torch.zeros(6, 6, dtype=torch.bool)
    for query in range(6):
        for key in range(6):
            expected[query, key] = (query // 3 == key // 3) and (key % 3 <= query % 3)
    assert torch.equal(own, expected)


def test_tree_walk_is_lossless_monte_carlo():
    """The first D+1 output tokens must follow the target law exactly (here: in TV)."""
    torch.manual_seed(71)
    num_chains, depth, vocab = 2, 2, 3
    target = {prefix: torch.softmax(torch.randn(vocab) * 1.3, dim=-1)
              for t in range(depth + 2) for prefix in product(range(vocab), repeat=t)}
    draft = torch.softmax(torch.randn(depth, vocab) * 1.3, dim=-1)
    generator = torch.Generator().manual_seed(73)

    def exact(seq):
        prob = 1.0
        for t in range(len(seq)):
            prob *= float(target[seq[:t]][seq[t]])
        return prob

    counts = {}
    trials = 30000
    for _ in range(trials):
        chains = torch.multinomial(draft, num_chains, replacement=True,
                                   generator=generator).T
        target_probs = torch.stack([
            torch.stack([target[tuple(chains[c, :s].tolist())] for s in range(depth + 1)])
            for c in range(num_chains)
        ])
        chain, accepted, next_token = tree_speculative_walk(
            chains, draft, target_probs, generator=generator
        )
        seq = tuple(chains[chain, :accepted].tolist()) + (int(next_token),)
        while len(seq) < depth + 1:                      # later rounds are lossless too
            seq = seq + (int(torch.multinomial(target[seq], 1, generator=generator)),)
        seq = seq[:depth + 1]
        counts[seq] = counts.get(seq, 0) + 1

    tv = 0.5 * sum(abs(counts.get(seq, 0) / trials - exact(seq))
                   for seq in product(range(vocab), repeat=depth + 1))
    assert tv < 0.025, tv


def test_tree_walk_with_identical_draft_and_target_accepts_everything():
    depth, vocab = 3, 4
    probs = torch.softmax(torch.randn(depth, vocab), dim=-1)
    chains = torch.multinomial(probs, 3, replacement=True).T
    # Target conditionals equal to the proposal for every chain: always accepted.
    target_probs = torch.cat([probs, probs[:1]], dim=0).unsqueeze(0).expand(3, -1, -1)
    _, accepted, _ = tree_speculative_walk(chains, probs, target_probs)
    assert accepted == depth
