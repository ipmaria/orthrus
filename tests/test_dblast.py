"""DBLAST re-implementation (src/dblast.py, loss_kl.fused_linear_dblast). Torch-only."""

from __future__ import annotations

from itertools import product

import torch
import torch.nn.functional as F

from src.dblast import (
    al_objective_terms,
    branch_hidden,
    category_log_prior,
    filtered_log_probs,
    greedy_branch_step,
)
from src.train.loss import IGNORE_INDEX
from src.train.loss_kl import fused_linear_dblast


def _reference_al(log_q_branch, log_prior, log_p, valid_len, tau):
    """Paper Eqs. 14-16 written out with Python loops for one block."""
    depth = valid_len
    log_q_prefix, log_r, accept = [], [], []
    for l in range(depth):
        log_q_prefix.append(torch.logsumexp(log_prior + log_q_branch[:l + 1].sum(0), 0))
        log_r.append(log_q_prefix[l] - log_p[:l + 1].sum())
        previous = log_q_prefix[l - 1] if l > 0 else torch.zeros(())
        accept.append(torch.exp(torch.clamp(log_p[l] - (log_q_prefix[l] - previous), max=0.0)))
    stop = depth - 1
    for l in range(depth):
        if float(log_r[l]) < torch.log(torch.tensor(tau)):
            stop = l
            break
    objective = torch.zeros(())
    for l in range(stop + 1):
        expected = sum(torch.stack(accept[:i + 1]).prod() for i in range(l + 1))
        objective = objective + log_r[l] + torch.log(expected)
    return objective


def test_al_objective_matches_the_paper_formula():
    torch.manual_seed(1)
    blocks, depth, K = 3, 5, 3
    log_q_branch = F.log_softmax(torch.randn(blocks, depth, K, 7), -1)[..., 0]
    log_prior = F.log_softmax(torch.randn(blocks, K), -1)
    log_p = torch.log(torch.rand(blocks, depth) * 0.9 + 0.05)
    prefix = torch.ones(blocks, depth, dtype=torch.bool)
    prefix[1, 3:] = False
    for tau in (0.1, 0.5, 1e-6):
        objective, _, _, _ = al_objective_terms(log_q_branch, log_prior, log_p, prefix, tau)
        for b in range(blocks):
            n = int(prefix[b].sum())
            expected = _reference_al(log_q_branch[b, :n], log_prior[b], log_p[b, :n], n, tau)
            assert torch.allclose(objective[b], expected, atol=1e-5), (tau, b)


def _naive_dblast(kind, weight, student, teacher, labels, block_size, K, heads, tau,
                  temperature, top_k, top_p):
    num_blocks = labels.shape[0] // block_size
    hidden = student.reshape(num_blocks, block_size, -1)[:, :-1]
    teacher = teacher.reshape(num_blocks, block_size, -1)[:, :-1]
    labels = labels.reshape(num_blocks, block_size)[:, :-1]
    total, kept = torch.zeros(()), 0
    for b in range(num_blocks):
        valid = labels[b] != IGNORE_INDEX
        n = int(valid.to(torch.int64).cumprod(0).sum())
        if n == 0:
            continue
        log_p = filtered_log_probs(teacher[b] @ weight.T, temperature, top_k, top_p)
        log_p = log_p.gather(-1, labels[b].clamp_min(0)[:, None])[:, 0]
        n = min(n, int(torch.isfinite(log_p).to(torch.int64).cumprod(0).sum()))
        if n == 0:
            continue
        branches = branch_hidden(hidden[b], heads[0], heads[1], K)
        log_q = F.log_softmax(branches @ weight.T / temperature, -1)
        log_q = log_q.gather(-1, labels[b].clamp_min(0)[:, None, None].expand(-1, K, 1))[..., 0]
        log_prior = category_log_prior(hidden[b, 0], heads[2], heads[3], K)
        if kind == "al":
            total = total - _reference_al(log_q[:n], log_prior, log_p[:n], n, tau)
        else:
            total = total - torch.logsumexp(log_prior + log_q[:n].sum(0), 0)
        kept += 1
    return total / kept


def test_dblast_losses_match_reference_value_and_gradient():
    torch.manual_seed(2)
    vocab, hidden, block_size, blocks, K = 13, 6, 5, 4, 3
    weight = torch.randn(vocab, hidden)
    teacher = torch.randn(blocks * block_size, hidden)
    labels = torch.randint(0, vocab, (blocks, block_size))
    labels[:, -1] = IGNORE_INDEX
    labels[2, 2] = IGNORE_INDEX
    labels = labels.reshape(-1)
    for kind in ("al", "nll"):
        for top_k, top_p in ((0, 1.0), (vocab, 0.999)):
            torch.manual_seed(3)
            student = torch.randn(blocks * block_size, hidden, requires_grad=True)
            heads = [torch.randn(K * hidden, hidden) * 0.3, torch.randn(K * hidden) * 0.1,
                     torch.randn(K, hidden), torch.randn(K)]
            heads = [h.requires_grad_(True) for h in heads]
            got = fused_linear_dblast(kind, weight, student, teacher, labels, block_size, K,
                                      *heads, tau=0.3, temperature=0.8, target_top_k=top_k,
                                      target_top_p=top_p, chunk_size=7)
            ref_student = student.detach().clone().requires_grad_(True)
            ref_heads = [h.detach().clone().requires_grad_(True) for h in heads]
            expected = _naive_dblast(kind, weight, ref_student, teacher, labels, block_size, K,
                                     ref_heads, 0.3, 0.8, top_k, top_p)
            assert torch.allclose(got, expected, atol=1e-4), (kind, got, expected)
            got.backward()
            expected.backward()
            assert torch.allclose(student.grad, ref_student.grad, atol=1e-4), kind
            for h, r in zip(heads, ref_heads):
                assert torch.allclose(h.grad, r.grad, atol=1e-4), kind


def test_k1_nll_is_the_independent_block_nll():
    torch.manual_seed(4)
    vocab, hidden, block_size, blocks = 11, 5, 4, 3
    weight = torch.randn(vocab, hidden)
    student = torch.randn(blocks * block_size, hidden)
    labels = torch.randint(0, vocab, (blocks, block_size))
    labels[:, -1] = IGNORE_INDEX
    got = fused_linear_dblast("nll", weight, student, torch.randn_like(student),
                              labels.reshape(-1), block_size, 1)
    log_q = F.log_softmax(student @ weight.T, -1).reshape(blocks, block_size, vocab)
    expected = -log_q[:, :-1].gather(-1, labels[:, :-1, None])[..., 0].sum(-1).mean()
    assert torch.allclose(got, expected, atol=1e-5)


def test_filter_matches_generate_sample_rule():
    torch.manual_seed(5)
    logits = torch.randn(6, 50) * 2
    for top_k, top_p in ((0, 1.0), (20, 1.0), (0, 0.8), (20, 0.8)):
        reference = logits.clone() / 0.7
        if top_k > 0:
            v, _ = torch.topk(reference, top_k)
            reference[reference < v[..., [-1]]] = -float("inf")
        if top_p < 1.0:
            sorted_logits, idx = torch.sort(reference, descending=True)
            cumulative = torch.cumsum(F.softmax(sorted_logits, -1), -1)
            remove = cumulative > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = 0
            reference[remove.scatter(-1, idx, remove)] = -float("inf")
        assert torch.allclose(filtered_log_probs(logits, 0.7, top_k, top_p).exp(),
                              F.softmax(reference, -1), atol=1e-6)


def test_greedy_branch_decoder_is_lossless_monte_carlo():
    torch.manual_seed(6)
    K, depth, vocab, trials = 3, 2, 3, 40000
    table = {pre: torch.softmax(torch.randn(vocab) * 1.4, -1)
             for t in range(depth + 2) for pre in product(range(vocab), repeat=t)}
    branches = torch.randint(0, vocab, (K, depth))
    pi = torch.softmax(torch.randn(K), -1)
    generator = torch.Generator().manual_seed(7)
    counts = {}
    for _ in range(trials):
        chosen = int(torch.multinomial(pi, 1, generator=generator))
        draft = branches[chosen].tolist()
        target = torch.stack([table[tuple(draft[:s])] for s in range(depth + 1)])
        accepted, next_token = greedy_branch_step(branches, pi, target, chosen, generator)
        seq = draft[:accepted] + [int(next_token)]
        while len(seq) < depth + 1:
            seq.append(int(torch.multinomial(table[tuple(seq)], 1, generator=generator)))
        counts[tuple(seq[:depth + 1])] = counts.get(tuple(seq[:depth + 1]), 0) + 1
    tv = 0.0
    for seq in product(range(vocab), repeat=depth + 1):
        exact = 1.0
        for t in range(depth + 1):
            exact *= float(table[seq[:t]][seq[t]])
        tv += abs(counts.get(seq, 0) / trials - exact)
    assert 0.5 * tv < 0.02, 0.5 * tv
