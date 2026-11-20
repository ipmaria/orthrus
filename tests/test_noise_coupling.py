"""Noise-conditioned drafting (src/noise_coupling.py). Torch-only."""

from __future__ import annotations

from itertools import product

import math
import torch
import torch.nn.functional as F

from src.noise_coupling import (
    noise_conditioned_speculative_step,
    noise_features,
    posterior_gumbel_on_support,
    revealed_law,
    revealed_log_law_at,
    sample_gumbel,
)
from src.train.loss import IGNORE_INDEX
from src.train.loss_kl import fused_linear_conditioned_target_path


def _probs(vocab, scale, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(vocab, generator=generator) * scale, dim=-1)


def test_posterior_noise_reproduces_the_token_and_is_prior_distributed():
    vocab, n = 5, 200000
    generator = torch.Generator().manual_seed(1)
    p = _probs(vocab, 1.3, 2)
    log_p = p.log()
    tokens = torch.multinomial(p, n, replacement=True, generator=generator)
    support = torch.arange(vocab).expand(n, vocab)
    in_support = support == tokens.unsqueeze(-1)
    gumbel = posterior_gumbel_on_support(log_p.expand(n, vocab), in_support, generator)
    assert torch.equal((log_p + gumbel).argmax(dim=-1), tokens)
    euler = 0.5772156649
    assert torch.allclose(gumbel.mean(0), torch.full((vocab,), euler), atol=0.01)
    assert torch.allclose(gumbel.var(0), torch.full((vocab,), math.pi ** 2 / 6), atol=0.03)


def test_revealed_law_matches_monte_carlo_and_limits():
    vocab = 6
    generator = torch.Generator().manual_seed(3)
    p = _probs(vocab, 1.2, 4)
    support = torch.tensor([1, 4])
    for _ in range(3):
        revealed = sample_gumbel((2,), generator=generator)
        law = revealed_law(p, support, revealed)
        assert abs(float(law.sum()) - 1.0) < 1e-6
        noise = sample_gumbel((200000, vocab), generator=generator)
        noise[:, support] = revealed
        empirical = torch.bincount((p.log() + noise).argmax(-1), minlength=vocab) / 200000
        assert torch.allclose(law, empirical, atol=0.006), (law, empirical)

    # Nothing revealed: the plain target. Everything revealed: the exact argmax.
    none = revealed_law(p, torch.zeros(0, dtype=torch.long), torch.zeros(0))
    assert torch.allclose(none, p, atol=1e-6)
    everything = torch.arange(vocab)
    g = sample_gumbel((vocab,), generator=generator)
    assert torch.allclose(revealed_law(p, everything, g),
                          F.one_hot((p.log() + g).argmax(), vocab).float(), atol=1e-6)


def test_revealed_law_averages_back_to_the_target():
    vocab, n = 5, 100000
    generator = torch.Generator().manual_seed(5)
    p = _probs(vocab, 1.5, 6)
    support = torch.tensor([0, 2, 3]).expand(n, 3)
    laws = revealed_law(p.expand(n, vocab), support, sample_gumbel((n, 3), generator=generator))
    assert torch.allclose(laws.mean(0), p, atol=0.004)


def test_log_law_at_token_agrees_with_full_law():
    vocab, n = 7, 2000
    generator = torch.Generator().manual_seed(7)
    p = _probs(vocab, 1.0, 8).expand(n, vocab)
    tokens = torch.multinomial(p[0], n, replacement=True, generator=generator)
    support = torch.tensor([2, 5, 6]).expand(n, 3)
    in_support = support == tokens.unsqueeze(-1)
    log_p_support = p.log().gather(-1, support)
    gumbel = posterior_gumbel_on_support(log_p_support, in_support, generator)
    got = revealed_log_law_at(log_p_support, gumbel, p.log().gather(-1, tokens[:, None])[:, 0],
                              in_support)
    full = revealed_law(p, support, gumbel).gather(-1, tokens[:, None])[:, 0].log()
    assert torch.allclose(got, full, atol=1e-4)


def test_conditioned_target_path_matches_dense_reference():
    torch.manual_seed(9)
    vocab, hidden, block_size, blocks = 11, 6, 4, 3
    weight = torch.randn(vocab, hidden)
    student = torch.randn(blocks * block_size, hidden, requires_grad=True)
    labels = torch.randint(0, vocab, (blocks, block_size))
    labels[:, -1] = IGNORE_INDEX
    labels[1, 2] = IGNORE_INDEX
    labels = labels.reshape(-1)
    log_target = torch.log(torch.rand(blocks * block_size) * 0.9 + 0.05)
    loss = fused_linear_conditioned_target_path(weight, student, log_target, labels,
                                                block_size, chunk_size=3)
    reference_student = student.detach().clone().requires_grad_(True)
    log_q = F.log_softmax(reference_student @ weight.T, -1)
    total, kept = 0.0, 0
    for b in range(blocks):
        survival, value = 1.0, 0.0
        for j in range(block_size - 1):
            i = b * block_size + j
            if labels[i] == IGNORE_INDEX:
                break
            survival = survival * torch.exp(torch.clamp(log_q[i, labels[i]] - log_target[i],
                                                        max=0.0))
            value = value + survival
        if not isinstance(value, float):
            total, kept = total + value, kept + 1
    expected = -total / kept
    assert torch.allclose(loss, expected, atol=1e-5)
    loss.backward()
    expected.backward()
    assert torch.allclose(student.grad, reference_student.grad, atol=1e-5)


def test_noise_conditioned_round_is_lossless_with_a_causal_drafter():
    """A full round with a drafter that depends on the revealed noise of positions <= i."""
    torch.manual_seed(13)
    vocab, depth, top_m, trials = 3, 2, 2, 40000
    p0 = _probs(vocab, 1.5, 14)
    transition = torch.softmax(torch.randn(vocab, vocab) * 1.5, dim=-1)
    cond = lambda prefix: p0 if not prefix else transition[prefix[-1]]
    support = torch.tensor([[0, 2], [1, 2]])
    theta = torch.randn(depth, vocab)
    coupling = torch.randn(depth, depth, top_m) * 1.5
    coupling = coupling * (torch.arange(depth)[None, :, None] <= torch.arange(depth)[:, None, None])

    counts = {}
    for _ in range(trials):
        gumbel = sample_gumbel((depth, top_m))
        logits = theta.clone()
        for s in range(depth):
            for j in range(depth):
                logits[s, support[j]] += coupling[s, j] * gumbel[j]
        q = torch.softmax(logits, dim=-1)
        draft = torch.multinomial(q, 1).squeeze(-1)
        target = torch.stack([cond(draft[:s].tolist()) for s in range(depth + 1)])
        accepted, next_token = noise_conditioned_speculative_step(draft, q, target, support,
                                                                  gumbel)
        seq = draft[:accepted].tolist() + [int(next_token)]
        while len(seq) < depth + 1:
            seq.append(int(torch.multinomial(cond(seq), 1)))
        key = tuple(seq[:depth + 1])
        counts[key] = counts.get(key, 0) + 1

    def exact(seq):
        prob = 1.0
        for t in range(len(seq)):
            prob *= float(cond(list(seq[:t]))[seq[t]])
        return prob

    tv = 0.5 * sum(abs(counts.get(seq, 0) / trials - exact(seq))
                   for seq in product(range(vocab), repeat=depth + 1))
    assert tv < 0.02, tv


def test_noise_features_shape():
    embed = torch.randn(13, 4)
    support = torch.tensor([[1, 2, 3], [4, 5, 6]])
    features = noise_features(embed, support, sample_gumbel((2, 3)))
    assert features.shape == (2, 8)
    assert torch.isfinite(features).all()
