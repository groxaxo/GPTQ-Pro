"""Opt-in GPTQ-initialized block reconstruction with unchanged V3 INT4 storage.

Independent implementation inspired by SignRound (arXiv:2309.05516). This is
not the AutoRound package or EXL3 encoding. Starts from existing packed integer
codes, optimizes offsets/scales with signed gradients, and selects on DISJOINT
validation examples. The baseline is always a candidate. Final test data must
not be passed to this optimizer. No model, global precision flag, or file mutation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from ..utils.gptq_pro_contract import pack_int4, unpack_int4, validate_v3_layout


@dataclass(frozen=True)
class PackedLinear:
    """Canonical unsigned [N,K] codes and stored FP16 [N,G] scales, zero=8."""

    codes: torch.Tensor
    scales: torch.Tensor
    group_size: int

    def validate(self) -> None:
        if self.codes.ndim != 2 or not self.codes.numel():
            raise ValueError("codes must be nonempty [N,K]")
        n, k = self.codes.shape
        if (
            isinstance(self.group_size, bool)
            or self.group_size not in (16, 32, 64, 128, 256, 512, 1024)
            or k % self.group_size
            or n % 8
            or k % 16
        ):
            raise ValueError("refinement requires aligned sequential V3 groups")
        if self.codes.dtype not in (torch.uint8, torch.int16, torch.int32, torch.int64):
            raise ValueError("codes must be integers")
        if self.codes.device != self.scales.device or self.codes.device.type == "meta":
            raise ValueError("codes/scales must be materialized on one device")
        if bool(((self.codes < 0) | (self.codes > 15)).any()):
            raise ValueError("codes out of INT4 range")
        if (
            self.scales.shape != (n, k // self.group_size)
            or self.scales.dtype != torch.float16
        ):
            raise ValueError("scales must be stored FP16 [N,K/group_size]")
        if not bool(torch.isfinite(self.scales).all() & (self.scales >= 0).all()):
            raise ValueError("scales must be finite and nonnegative")
        if not bool(torch.isfinite(self.weight()).all()):
            raise ValueError("FP16 dequantization overflow")

    def weight(self) -> torch.Tensor:
        return (
            (self.codes.float() - 8)
            * self.scales.float().repeat_interleave(self.group_size, 1)
        ).half()

    def clone(self) -> PackedLinear:
        return PackedLinear(
            self.codes.detach().clone(), self.scales.detach().clone(), self.group_size
        )

    def to(self, device) -> PackedLinear:
        return PackedLinear(
            self.codes.to(device), self.scales.to(device), self.group_size
        )

    def pack(self) -> dict[str, torch.Tensor]:
        self.validate()
        n, k = self.codes.shape
        tensors = {
            "qweight": pack_int4(self.codes.T.contiguous()),
            "scales": self.scales.T.contiguous(),
            "qzeros": pack_int4(
                torch.full(
                    (k // self.group_size, n),
                    8,
                    dtype=torch.int32,
                    device=self.codes.device,
                ),
                axis=1,
            ),
            "g_idx": torch.arange(k, dtype=torch.int32, device=self.codes.device)
            // self.group_size,
        }
        validate_v3_layout(**tensors, group_size=self.group_size, qzero_format=2)
        return tensors

    @classmethod
    def from_packed(cls, *, qweight, scales, qzeros, g_idx, group_size, qzero_format=2):
        validate_v3_layout(
            qweight,
            scales,
            group_size,
            qzeros=qzeros,
            g_idx=g_idx,
            qzero_format=qzero_format,
        )
        result = cls(
            unpack_int4(qweight).T.contiguous().to(torch.uint8),
            scales.T.contiguous().clone(),
            group_size,
        )
        result.validate()
        return result

    @classmethod
    def from_solver(cls, quantized_weight, scales, zeros, g_idx, group_size):
        """Use the original packer's integer equation before casting scales.

        quantized_weight is the GPTQ result, never the original BF16 weights.
        Importing from packed tensors is preferred when a checkpoint exists.
        """
        n, k = quantized_weight.shape
        expected = torch.arange(k, device=g_idx.device, dtype=g_idx.dtype) // group_size
        if not torch.equal(g_idx, expected) or not bool((zeros == 8).all()):
            raise ValueError(
                "only sequential groups and zero-point eight are supported"
            )
        if scales.shape != (n, k // group_size) or not bool(
            torch.isfinite(scales).all() & (scales > 0).all()
        ):
            raise ValueError("invalid solver scales")
        expanded = scales.float().repeat_interleave(group_size, 1)
        codes = torch.round((quantized_weight.float() + 8 * expanded) / expanded)
        if not bool(torch.isfinite(codes).all()) or bool(
            ((codes < 0) | (codes > 15)).any()
        ):
            raise ValueError("solver result cannot be represented without clipping")
        result = cls(codes.to(torch.uint8), scales.half().clone(), group_size)
        result.validate()
        return result


@dataclass(frozen=True)
class RefinementConfig:
    steps: int = 50
    batch_rows: int = 128
    eval_every: int = 5
    round_lr: float = 0.05
    scale_lr: float = 0.002
    max_code_delta: int = 1
    max_scale_ratio: float = 1.25
    tune_rounding: bool = True
    tune_scales: bool = True
    min_train_rows: int = 128
    min_validation_rows: int = 32
    min_relative_gain: float = 0.0
    seed: int = 20261015

    def validate(self):
        for name in (
            "steps",
            "batch_rows",
            "eval_every",
            "max_code_delta",
            "min_train_rows",
            "min_validation_rows",
            "seed",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if name in ("steps", "seed") else 1)
            ):
                raise ValueError(f"invalid {name}")
        if self.max_code_delta > 2:
            raise ValueError("max_code_delta is bounded to two INT4 bins")
        for name in ("round_lr", "scale_lr", "max_scale_ratio", "min_relative_gain"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"nonfinite {name}")
        if not (
            0 < self.round_lr <= 1
            and 0 < self.scale_lr <= 0.1
            and 1 <= self.max_scale_ratio <= 2
            and 0 <= self.min_relative_gain < 1
        ):
            raise ValueError("invalid optimization bound")
        if not isinstance(self.tune_rounding, bool) or not isinstance(
            self.tune_scales, bool
        ):
            raise TypeError("tuning flags must be booleans")


def validate_splits(train_ids, validation_ids, test_ids=()):
    """IDs must identify whole conversations/documents, NOT adjacent token rows."""
    for values in (train_ids, validation_ids, test_ids):
        if not isinstance(values, (list, tuple)):
            raise TypeError(
                "split IDs must be lists or tuples of whole-document identities"
            )
    splits = (tuple(train_ids), tuple(validation_ids), tuple(test_ids))
    for index, values in enumerate(splits):
        if index < 2 and not values:
            raise ValueError("train and validation need example provenance IDs")
        if any(not isinstance(v, str) or not v for v in values) or len(
            set(values)
        ) != len(values):
            raise ValueError("IDs must be nonempty unique strings within each split")
    if any(set(splits[a]) & set(splits[b]) for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("example overlap across calibration/validation/test splits")


def _finite_tensor(value, name):
    if (
        value.device.type == "meta"
        or not value.numel()
        or not bool(torch.isfinite(value).all())
    ):
        raise FloatingPointError(f"nonfinite or empty {name}")


def linear_forward(x, weights):
    if len(weights) != 1:
        raise ValueError("linear reconstruction needs exactly one weight")
    return F.linear(x, next(iter(weights.values())))


def swiglu_forward(x, weights):
    if set(weights) != {"gate_proj", "up_proj", "down_proj"}:
        raise ValueError("SwiGLU reconstruction needs gate/up/down weights")
    return F.linear(
        F.silu(F.linear(x, weights["gate_proj"])) * F.linear(x, weights["up_proj"]),
        weights["down_proj"],
    )


@torch.no_grad()
def reconstruction_error(
    source_weights: Mapping[str, torch.Tensor],
    candidate: Mapping[str, PackedLinear],
    inputs: torch.Tensor,
    forward: Callable = linear_forward,
    chunk_rows: int = 128,
) -> float:
    """NMSE in FP32 accumulation, with stored FP16 weight operands.

    Protected components retain their exact source values. This measures isolated
    block error, not end-to-end KL, BF16 kernel parity, or CUDA reduction order.
    """
    if (
        isinstance(chunk_rows, bool)
        or not isinstance(chunk_rows, int)
        or chunk_rows < 1
    ):
        raise ValueError("chunk_rows must be positive")
    _finite_tensor(inputs, "evaluation inputs")
    if not set(candidate) <= set(source_weights):
        raise ValueError("candidate has unknown source components")
    source = {k: v.float() for k, v in source_weights.items()}
    weights = {
        k: candidate[k].weight().float() if k in candidate else v
        for k, v in source.items()
    }
    numerator, denominator = 0.0, 0.0
    for x in inputs.split(chunk_rows):
        reference = forward(x.float(), source)
        actual = forward(x.float(), weights)
        _finite_tensor(reference, "reference output")
        _finite_tensor(actual, "candidate output")
        numerator += (actual.double() - reference.double()).square().sum().item()
        denominator += reference.double().square().sum().item()
    if denominator <= 0:
        raise ValueError("zero-energy reference output")
    return numerator / denominator


@torch.inference_mode(False)
@torch.enable_grad()
def refine_block(
    source_weights: Mapping[str, torch.Tensor],
    baseline: Mapping[str, PackedLinear],
    train_inputs: torch.Tensor,
    validation_inputs: torch.Tensor,
    *,
    train_ids,
    validation_ids,
    config: RefinementConfig | None = None,
    forward: Callable = linear_forward,
):
    """Return a new candidate and diagnostics; never modify source or baseline.

    ``baseline`` may be a subset for protected BF16 projections. Bounded one-block
    operation; explicitly decline under-covered training/validation inputs.
    The caller is responsible for truthful document IDs and an untouched test set.
    """
    config = config or RefinementConfig()
    config.validate()
    validate_splits(train_ids, validation_ids)
    if not baseline or not set(baseline) <= set(source_weights):
        raise ValueError("candidate components must be a nonempty source subset")
    for name, weight in source_weights.items():
        _finite_tensor(weight, name)
        if weight.ndim != 2 or weight.device != train_inputs.device:
            raise ValueError("source weights must be matrices on the input device")
    for name, packed in baseline.items():
        packed.validate()
        if (
            packed.codes.shape != source_weights[name].shape
            or packed.codes.device != train_inputs.device
        ):
            raise ValueError("baseline/source shape or device mismatch")
        # Native V3 legitimately supports zero stored scales for tiny BF16
        # source groups. Freeze those groups, but refuse to hide material weights.
        grouped = (
            source_weights[name]
            .detach()
            .float()
            .reshape(packed.scales.shape[0], packed.scales.shape[1], packed.group_size)
        )
        if bool(((packed.scales == 0) & (grouped.abs().amax(-1) > 2**-20)).any()):
            raise ValueError(
                "zero-scale group masks material source weights; repair the baseline first"
            )
    for label, value in (("train", train_inputs), ("validation", validation_inputs)):
        _finite_tensor(value, label)
        if value.ndim != 2 or value.device != train_inputs.device:
            raise ValueError("inputs must be [rows,features] on one device")
    if train_inputs.shape[1] != validation_inputs.shape[1]:
        raise ValueError("train/validation input widths differ")
    # Clone inference-mode captures before building an autograd graph.
    source = {k: v.detach().clone().float() for k, v in source_weights.items()}
    train = train_inputs.detach().clone().float()
    validation = validation_inputs.detach().clone().float()
    best = {k: v.clone() for k, v in baseline.items()}
    baseline_error = reconstruction_error(
        source, best, validation, forward, config.batch_rows
    )
    report = {
        "algorithm": "gptq-initialized-sign-refinement-v1",
        "config": asdict(config),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "protected_components": sorted(set(source) - set(baseline)),
        "frozen_zero_scale_groups": {
            k: int((v.scales == 0).sum()) for k, v in baseline.items()
        },
        "baseline_validation_nmse": baseline_error,
        "selected_validation_nmse": baseline_error,
        "selected_step": 0,
        "accepted": False,
        "history": [{"step": 0, "validation_nmse": baseline_error}],
        "objective": "block-NMSE, FP32 accumulation, FP16 stored/dequant operands; not full-model quality",
    }
    if (
        len(train) < config.min_train_rows
        or len(validation) < config.min_validation_rows
    ):
        report["status"] = "blocked_undercoverage"
        return best, report
    if not config.steps or not (config.tune_rounding or config.tune_scales):
        report["status"] = "baseline_only"
        return best, report
    deltas = {
        k: torch.nn.Parameter(
            torch.zeros_like(v.codes, dtype=torch.float32),
            requires_grad=config.tune_rounding,
        )
        for k, v in baseline.items()
    }
    logs = {
        k: torch.nn.Parameter(
            torch.zeros_like(v.scales, dtype=torch.float32),
            requires_grad=config.tune_scales,
        )
        for k, v in baseline.items()
    }
    bound = math.log(config.max_scale_ratio)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    def candidate(hard=False):
        result = dict(source)
        packed = {}
        for name, original in baseline.items():
            raw = (original.codes.float() + deltas[name]).clamp(0, 15)
            integers = raw.round()
            ste_codes = raw + (integers - raw).detach()
            scales = original.scales.float() * logs[name].clamp(-bound, bound).exp()
            stored = scales.half().float()
            if not bool(torch.isfinite(stored).all() & (stored >= 0).all()) or bool(
                ((original.scales > 0) & (stored == 0)).any()
            ):
                raise FloatingPointError("invalid stored scales during refinement")
            ste_scales = scales + (stored - scales).detach()
            weight = (ste_codes - 8) * ste_scales.repeat_interleave(
                original.group_size, 1
            )
            # Include FP16 dequant-product rounding in both forward and selection.
            result[name] = weight + (weight.half().float() - weight).detach()
            if hard:
                packed[name] = PackedLinear(
                    integers.detach().to(torch.uint8),
                    stored.detach().half(),
                    original.group_size,
                )
        return packed if hard else result

    try:
        for step in range(1, config.steps + 1):
            for parameter in (*deltas.values(), *logs.values()):
                parameter.grad = None
            indices = torch.randperm(len(train), generator=generator)[
                : config.batch_rows
            ].to(train.device)
            x = train[indices]
            with torch.no_grad():
                target = forward(x, source)
            actual = forward(x, candidate())
            loss = (actual - target).square().mean() / target.square().mean().clamp_min(
                1e-20
            )
            _finite_tensor(loss, "training loss")
            loss.backward()
            with torch.no_grad():
                for parameters, lr, clip in (
                    (deltas, config.round_lr, float(config.max_code_delta)),
                    (logs, config.scale_lr, bound),
                ):
                    for parameter in parameters.values():
                        if parameter.requires_grad:
                            if parameter.grad is None:
                                raise RuntimeError("missing reconstruction gradient")
                            _finite_tensor(parameter.grad, "gradient")
                            parameter.add_(parameter.grad.sign(), alpha=-lr).clamp_(
                                -clip, clip
                            )
                if step % config.eval_every == 0 or step == config.steps:
                    proposed = candidate(hard=True)
                    for value in proposed.values():
                        value.validate()
                    error = reconstruction_error(
                        source, proposed, validation, forward, config.batch_rows
                    )
                    report["history"].append(
                        {
                            "step": step,
                            "validation_nmse": error,
                            "train_nmse": loss.item(),
                        }
                    )
                    if error < report[
                        "selected_validation_nmse"
                    ] and error < baseline_error * (1 - config.min_relative_gain):
                        best = {k: v.clone() for k, v in proposed.items()}
                        report.update(
                            selected_validation_nmse=error,
                            selected_step=step,
                            accepted=True,
                        )
    except FloatingPointError as exc:
        # Resource/runtime/programming errors propagate; only numerical candidates
        # are rejected. The immutable best-so-far remains available and marked.
        report["status"] = "numerical_candidate_rejected"
        report["error"] = str(exc)
    else:
        report["status"] = "improved" if report["accepted"] else "baseline_retained"
    report["changed_codes"] = {
        k: int((best[k].codes != baseline[k].codes).sum()) for k in best
    }
    report["changed_scales"] = {
        k: int((best[k].scales != baseline[k].scales).sum()) for k in best
    }
    for value in best.values():
        value.pack()  # Exact packing contract is a mandatory exit gate.
    return best, report
