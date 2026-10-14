"""Bounded, response-token KL/NLL accumulation against a full-precision reference.

The caller streams aligned logits/next-token targets in sequence order. Nothing is
saved except aggregate numbers. A mask must select RESPONSE target tokens, not
prompt tokens. No full-vocabulary probability archive is required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class StreamingLanguageMetrics:
    token_chunk: int = 64
    vocab_chunk: int = 4096
    tokens: int = 0
    kl_sum: float = 0.0
    reference_nll_sum: float = 0.0
    candidate_nll_sum: float = 0.0
    argmax_matches: int = 0

    def __post_init__(self):
        for value in (self.token_chunk, self.vocab_chunk):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("chunk sizes must be positive integers")

    @torch.no_grad()
    def update(self, reference_logits, candidate_logits, targets, response_mask):
        if (
            reference_logits.shape != candidate_logits.shape
            or reference_logits.ndim < 2
        ):
            raise ValueError(
                "reference/candidate logits must have the same [...,vocab] shape"
            )
        shape, vocab = reference_logits.shape[:-1], reference_logits.shape[-1]
        if (
            targets.shape != shape
            or response_mask.shape != shape
            or response_mask.dtype != torch.bool
        ):
            raise ValueError("targets and boolean response mask must align to logits")
        if targets.dtype not in (torch.int32, torch.int64) or vocab < 2:
            raise ValueError("invalid targets or vocabulary")
        device = reference_logits.device
        if any(t.device != device for t in (candidate_logits, targets, response_mask)):
            raise ValueError("all metric tensors must share a device")
        ref = reference_logits.reshape(-1, vocab)
        cand = candidate_logits.reshape(-1, vocab)
        labels, mask = targets.reshape(-1), response_mask.reshape(-1)
        # Validate before changing accumulators; a bad batch cannot partially count.
        for r, c in zip(ref.split(self.token_chunk), cand.split(self.token_chunk)):
            if not bool(torch.isfinite(r).all() & torch.isfinite(c).all()):
                raise FloatingPointError("nonfinite logits")
        active = mask & (labels != -100)
        if bool((((labels < 0) | (labels >= vocab)) & active).any()):
            raise ValueError("active target is out of vocabulary")
        local_tokens, local_kl, local_rnll, local_cnll, local_matches = (
            0,
            0.0,
            0.0,
            0.0,
            0,
        )
        for start in range(0, len(ref), self.token_chunk):
            select = active[start : start + self.token_chunk]
            if not bool(select.any()):
                continue
            r, c = (
                ref[start : start + self.token_chunk][select],
                cand[start : start + self.token_chunk][select],
            )
            y = labels[start : start + self.token_chunk][select].long()
            # FP64 log-normalizers and sum-of-chunk KL avoid cancellation near zero.
            rz = torch.full((len(r),), -torch.inf, dtype=torch.float64, device=device)
            cz = rz.clone()
            for lo in range(0, vocab, self.vocab_chunk):
                rz = torch.logaddexp(
                    rz, torch.logsumexp(r[:, lo : lo + self.vocab_chunk].double(), -1)
                )
                cz = torch.logaddexp(
                    cz, torch.logsumexp(c[:, lo : lo + self.vocab_chunk].double(), -1)
                )
            divergence = torch.zeros_like(rz)
            for lo in range(0, vocab, self.vocab_chunk):
                rp = r[:, lo : lo + self.vocab_chunk].double() - rz[:, None]
                cp = c[:, lo : lo + self.vocab_chunk].double() - cz[:, None]
                divergence += (rp.exp() * (rp - cp)).sum(-1)
            if bool((divergence < -1e-9).any()):
                raise FloatingPointError("negative KL exceeds numerical tolerance")
            rows = torch.arange(len(y), device=device)
            local_kl += divergence.clamp_min(0).sum().item()
            local_rnll += (rz - r[rows, y].double()).sum().item()
            local_cnll += (cz - c[rows, y].double()).sum().item()
            local_matches += int((r.argmax(-1) == c.argmax(-1)).sum())
            local_tokens += len(y)
        if not all(math.isfinite(v) for v in (local_kl, local_rnll, local_cnll)):
            raise FloatingPointError("nonfinite metric aggregate")
        self.tokens += local_tokens
        self.kl_sum += local_kl
        self.reference_nll_sum += local_rnll
        self.candidate_nll_sum += local_cnll
        self.argmax_matches += local_matches

    def result(self):
        if not self.tokens:
            raise ValueError("no response tokens evaluated")
        r, c = (
            self.reference_nll_sum / self.tokens,
            self.candidate_nll_sum / self.tokens,
        )
        return {
            "response_tokens": self.tokens,
            "kl_reference_to_candidate": self.kl_sum / self.tokens,
            "reference_nll": r,
            "candidate_nll": c,
            "delta_nll": c - r,
            "reference_perplexity": math.exp(r) if r < 700 else None,
            "candidate_perplexity": math.exp(c) if c < 700 else None,
            "argmax_agreement": self.argmax_matches / self.tokens,
            "alignment": "caller-supplied next-token targets and response-only mask",
        }

    def update_causal(
        self, reference_logits, candidate_logits, input_ids, response_token_mask
    ):
        """Convenience path for complete causal sequences, with explicit shift."""
        if input_ids.ndim < 1 or input_ids.shape[-1] < 2:
            raise ValueError("causal scoring needs at least two tokens")
        if (
            reference_logits.shape[:-1] != input_ids.shape
            or response_token_mask.shape != input_ids.shape
        ):
            raise ValueError("causal sequence dimensions disagree")
        self.update(
            reference_logits[..., :-1, :],
            candidate_logits[..., :-1, :],
            input_ids[..., 1:],
            response_token_mask[..., 1:],
        )
