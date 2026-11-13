import time
import os
import gc
import json
import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


def measure_tensor_size_mb(model: nn.Module) -> float:
    """
    Calculate the total size of all parameters and buffers in megabytes.
    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024.0 * 1024.0)


def measure_adapter_size_mb(model: nn.Module) -> float:
    """
    Calculate the size of adapter parameters in megabytes.
    """
    from .layer import AdaptiveRankLinear
    from .compact import CompactAdaptiveRankLinear

    adapter_bytes = 0
    for _, module in model.named_modules():
        if isinstance(module, AdaptiveRankLinear):
            for p in [module.A, module.B, module.c]:
                if p is not None:
                    adapter_bytes += p.nelement() * p.element_size()
        elif isinstance(module, CompactAdaptiveRankLinear):
            for p in [module.A, module.B, module.c]:
                if p is not None:
                    adapter_bytes += p.nelement() * p.element_size()

    return adapter_bytes / (1024.0 * 1024.0)


@torch.no_grad()
def benchmark_model_latency(
    model: nn.Module,
    batch_size: int = 1,
    seq_length: int = 128,
    warmup_runs: int = 15,
    benchmark_runs: int = 50,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """
    High-resolution benchmarking of model inference latency and throughput.
    
    Includes proper GPU synchronization (torch.cuda.synchronize) when running on CUDA.
    """
    device = torch.device(device)
    model.eval()
    model.to(device)

    # Generate synthetic input batch matching Transformer sequence classification signature
    input_ids = torch.randint(100, 1000, (batch_size, seq_length), dtype=torch.long, device=device)
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=device)
    batch = {"input_ids": input_ids, "attention_mask": attention_mask}

    is_cuda = (device.type == "cuda")

    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Warmup runs
    for _ in range(warmup_runs):
        _ = model(**batch)
        if is_cuda:
            torch.cuda.synchronize(device)

    # Benchmark runs
    latencies_ms = []
    for _ in range(benchmark_runs):
        if is_cuda:
            torch.cuda.synchronize(device)
        t_start = time.perf_counter()

        _ = model(**batch)

        if is_cuda:
            torch.cuda.synchronize(device)
        t_end = time.perf_counter()

        latencies_ms.append((t_end - t_start) * 1000.0)

    latencies = np.array(latencies_ms)
    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    median_lat = float(np.median(latencies))
    p90_lat = float(np.percentile(latencies, 90))
    p95_lat = float(np.percentile(latencies, 95))
    p99_lat = float(np.percentile(latencies, 99))
    min_lat = float(np.min(latencies))
    max_lat = float(np.max(latencies))

    throughput = float((batch_size / (mean_lat / 1000.0))) if mean_lat > 0 else 0.0

    peak_memory_mb = 0.0
    if is_cuda:
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))

    return {
        "batch_size": batch_size,
        "seq_length": seq_length,
        "warmup_runs": warmup_runs,
        "benchmark_runs": benchmark_runs,
        "device": str(device),
        "mean_latency_ms": round(mean_lat, 4),
        "std_latency_ms": round(std_lat, 4),
        "median_latency_ms": round(median_lat, 4),
        "p90_latency_ms": round(p90_lat, 4),
        "p95_latency_ms": round(p95_lat, 4),
        "p99_latency_ms": round(p99_lat, 4),
        "min_latency_ms": round(min_lat, 4),
        "max_latency_ms": round(max_lat, 4),
        "throughput_samples_per_sec": round(throughput, 2),
        "peak_gpu_memory_mb": round(peak_memory_mb, 2),
    }


def compare_inference_variants(
    gated_model: nn.Module,
    compact_model: Optional[nn.Module] = None,
    unpruned_model: Optional[nn.Module] = None,
    batch_sizes: List[int] = [1, 32],
    seq_length: int = 128,
    warmup_runs: int = 10,
    benchmark_runs: int = 30,
    device: Union[str, torch.device] = "cpu",
) -> List[Dict[str, Any]]:
    """
    Run comparative benchmarking between gated, compact, and unpruned variants.
    """
    variants = [("Gated DoRA (Soft Pruned)", gated_model)]
    if compact_model is not None:
        variants.append(("Compact DoRA (Physically Pruned)", compact_model))
    if unpruned_model is not None:
        variants.append(("Unpruned DoRA (Initial Rank)", unpruned_model))

    results = []
    for variant_name, model in variants:
        total_size_mb = measure_tensor_size_mb(model)
        adapter_size_mb = measure_adapter_size_mb(model)

        for bs in batch_sizes:
            logger.info(f"Benchmarking {variant_name} with batch_size={bs}...")
            bench_stats = benchmark_model_latency(
                model=model,
                batch_size=bs,
                seq_length=seq_length,
                warmup_runs=warmup_runs,
                benchmark_runs=benchmark_runs,
                device=device,
            )

            record = {
                "variant": variant_name,
                "total_model_size_mb": round(total_size_mb, 3),
                "adapter_size_mb": round(adapter_size_mb, 3),
                **bench_stats,
            }
            results.append(record)

    return results
