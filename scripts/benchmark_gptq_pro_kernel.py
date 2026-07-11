#!/usr/bin/env python3
"""Benchmark GPTQ-Pro's raw CUDA kernels without model-loading overhead.

The script measures AUTO, legacy, and applicable specialized paths for a set of
M values. It checks a column subset against a PyTorch FP32 reference using
FP16-dequantized weights, matching the kernel's numerical contract.

Example:

    python scripts/benchmark_gptq_pro_kernel.py \
        --m-values 1,4,8,16,64,256 --n 4096 --k 4096 --group-size 128 \
        --warmup 20 --iterations 100 --output kernel-results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from gptqmodel.utils.gptq_pro import ensure_gptq_pro_loaded


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m-values", type=parse_int_list, default=parse_int_list("1,4,8,16,64,256")
    )
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--check-columns", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    for name in ("n", "k", "group_size", "warmup", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.k % 8 != 0:
        parser.error("--k must be divisible by 8 for native GPTQ int32 packing")
    if args.group_size % 16 != 0:
        parser.error("--group-size must be a multiple of 16")
    if args.k % args.group_size != 0:
        parser.error("--k must be divisible by --group-size for this benchmark")
    if args.check_columns <= 0:
        parser.error("--check-columns must be greater than zero")
    return args


def make_problem(m: int, n: int, k: int, group_size: int, device: torch.device):
    activations = torch.randn((m, k), device=device, dtype=torch.float16)
    qweight_bytes = torch.randint(
        0,
        256,
        (k // 8, n, 4),
        device=device,
        dtype=torch.uint8,
    )
    qweight = qweight_bytes.view(torch.int32).squeeze(-1).contiguous()
    scales = (
        torch.rand((k // group_size, n), device=device, dtype=torch.float16) * 0.08
        + 0.001
    )
    return activations, qweight, scales


def dequantize_reference(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    k: int,
    group_size: int,
) -> torch.Tensor:
    shifts = torch.arange(
        0,
        32,
        4,
        device=qweight.device,
        dtype=torch.int32,
    ).view(1, 8, 1)
    values = (
        torch.bitwise_and(torch.bitwise_right_shift(qweight.unsqueeze(1), shifts), 0xF)
        .reshape(k, qweight.shape[1])
        .to(torch.float16)
        - 8
    )
    group_index = torch.arange(k, device=qweight.device) // group_size
    return values * scales.index_select(0, group_index)


def check_numerics(
    module,
    activations: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    mode: str,
    columns: int,
) -> dict[str, float]:
    columns = min(columns, qweight.shape[1])
    if columns >= 16:
        columns = max(16, columns - columns % 16)
    qweight_subset = qweight[:, :columns].contiguous()
    scales_subset = scales[:, :columns].contiguous()

    actual = module.gptq_pro_gemm(
        activations,
        qweight_subset,
        scales_subset,
        group_size,
        mode,
    )
    weights = dequantize_reference(
        qweight_subset,
        scales_subset,
        activations.shape[1],
        group_size,
    )
    expected = torch.matmul(activations.float(), weights.float()).to(torch.float16)
    difference = actual.float() - expected.float()
    maximum_absolute_error = difference.abs().max().item()
    mean_absolute_error = difference.abs().mean().item()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().reshape(1, -1),
        expected.float().reshape(1, -1),
    ).item()
    if not torch.isfinite(actual).all():
        raise RuntimeError(f"non-finite output from kernel mode {mode}")
    return {
        "columns": columns,
        "max_abs_error": maximum_absolute_error,
        "mean_abs_error": mean_absolute_error,
        "cosine_similarity": cosine,
    }


def benchmark_mode(
    module,
    activations: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    mode: str,
    warmup: int,
    iterations: int,
) -> dict[str, float | str]:
    for _ in range(warmup):
        module.gptq_pro_gemm(activations, qweight, scales, group_size, mode)
    torch.cuda.synchronize(activations.device)

    times_ms: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        module.gptq_pro_gemm(activations, qweight, scales, group_size, mode)
        end.record()
        end.synchronize()
        times_ms.append(float(start.elapsed_time(end)))

    median_ms = statistics.median(times_ms)
    mean_ms = statistics.fmean(times_ms)
    sorted_times = sorted(times_ms)
    p95_ms = sorted_times[min(len(sorted_times) - 1, int(len(sorted_times) * 0.95))]
    m, k = activations.shape
    n = qweight.shape[1]
    dense_operations = 2.0 * m * n * k
    tflops = dense_operations / (median_ms / 1000.0) / 1.0e12
    minimum_bytes = (
        activations.numel() * activations.element_size()
        + qweight.numel() * qweight.element_size()
        + scales.numel() * scales.element_size()
        + m * n * activations.element_size()
    )
    minimum_bandwidth_gbs = minimum_bytes / (median_ms / 1000.0) / 1.0e9
    return {
        "mode": mode,
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "p95_ms": p95_ms,
        "effective_dense_tflops": tflops,
        "minimum_bandwidth_gbs": minimum_bandwidth_gbs,
    }


def applicable_modes(m: int, n: int, k: int) -> list[str]:
    modes = ["auto", "legacy"]
    if m <= 4 and k % 8 == 0:
        modes.append("gemv")
    if n % 16 == 0 and k % 16 == 0:
        modes.append("ampere")
    return modes


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        raise SystemExit(f"compute capability 8.0+ is required, got {major}.{minor}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    module = ensure_gptq_pro_loaded(verbose=True)
    properties = torch.cuda.get_device_properties(device)
    report = {
        "created_unix": time.time(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": properties.name,
        "compute_capability": f"{major}.{minor}",
        "total_memory_bytes": properties.total_memory,
        "shape": {"n": args.n, "k": args.k, "group_size": args.group_size},
        "warmup": args.warmup,
        "iterations": args.iterations,
        "results": [],
    }

    print(
        f"GPTQ-Pro raw kernel benchmark on {properties.name} (sm_{major}{minor}), "
        f"N={args.n}, K={args.k}, group={args.group_size}"
    )
    for m in args.m_values:
        activations, qweight, scales = make_problem(
            m, args.n, args.k, args.group_size, device
        )
        for mode in applicable_modes(m, args.n, args.k):
            numerical = check_numerics(
                module,
                activations,
                qweight,
                scales,
                args.group_size,
                mode,
                args.check_columns,
            )
            timing = benchmark_mode(
                module,
                activations,
                qweight,
                scales,
                args.group_size,
                mode,
                args.warmup,
                args.iterations,
            )
            result = {"m": m, **timing, "numerical": numerical}
            report["results"].append(result)
            print(
                f"M={m:4d} mode={mode:7s} median={timing['median_ms']:.4f} ms "
                f"p95={timing['p95_ms']:.4f} ms "
                f"TFLOP/s={timing['effective_dense_tflops']:.2f} "
                f"cos={numerical['cosine_similarity']:.8f}"
            )
        del activations, qweight, scales
        torch.cuda.empty_cache()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
