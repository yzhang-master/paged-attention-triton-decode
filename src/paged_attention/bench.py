from __future__ import annotations

import platform
from datetime import datetime
from typing import Any

import torch

from paged_attention.cache import PagedKVCache
from paged_attention.kernel import paged_decode_attention
from paged_attention.reference import contiguous_attention


def cpu_smoke() -> dict[str, Any]:
    torch.manual_seed(71)
    lengths = [1, 5, 9, 14]
    cache = PagedKVCache(num_blocks=20, block_size=4, num_heads=3, head_dim=12)
    request_ids = list(range(len(lengths)))
    keys = []
    values = []
    for request_id, length in zip(request_ids, lengths, strict=True):
        cache.create_sequence(request_id)
        key = torch.randn(length, 3, 12)
        value = torch.randn(length, 3, 12)
        cache.append(request_id, key, value)
        keys.append(key)
        values.append(value)
    query = torch.randn(len(lengths), 3, 12)
    actual = paged_decode_attention(query, cache, request_ids)
    expected = contiguous_attention(query, keys, values)
    max_error = (actual - expected).abs().max().item()
    return {
        "status": "correctness_only",
        "device": "cpu",
        "triton_execution": "interpreter",
        "lengths": lengths,
        "max_absolute_error": max_error,
        "performance_data": False,
    }


def _cuda_milliseconds(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _gpu_case(
    batch_size: int,
    context_length: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, float | int]:
    device = torch.device("cuda")
    blocks_per_request = (context_length + block_size - 1) // block_size
    cache = PagedKVCache(
        batch_size * blocks_per_request,
        block_size,
        num_heads,
        head_dim,
        device=device,
    )
    request_ids = list(range(batch_size))
    keys = []
    values = []
    for request_id in request_ids:
        cache.create_sequence(request_id)
        key = torch.randn(context_length, num_heads, head_dim, device=device)
        value = torch.randn(context_length, num_heads, head_dim, device=device)
        cache.append(request_id, key, value)
        keys.append(key)
        values.append(value)
    query = torch.randn(batch_size, num_heads, head_dim, device=device)
    contiguous_keys = torch.stack(keys)
    contiguous_values = torch.stack(values)

    def contiguous() -> torch.Tensor:
        scores = torch.einsum("bhd,blhd->bhl", query, contiguous_keys) / head_dim**0.5
        probabilities = torch.softmax(scores, dim=-1)
        return torch.einsum("bhl,blhd->bhd", probabilities, contiguous_values)

    def paged() -> torch.Tensor:
        return paged_decode_attention(query, cache, request_ids)
    torch.testing.assert_close(paged(), contiguous(), atol=2e-4, rtol=2e-4)
    paged_ms = _cuda_milliseconds(paged, warmup, iterations)
    contiguous_ms = _cuda_milliseconds(contiguous, warmup, iterations)
    return {
        "batch_size": batch_size,
        "context_length": context_length,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "block_size": block_size,
        "paged_milliseconds": paged_ms,
        "contiguous_milliseconds": contiguous_ms,
        "speedup": contiguous_ms / paged_ms,
    }


def benchmark(warmup: int = 10, iterations: int = 50) -> dict[str, Any]:
    result: dict[str, Any] = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    if not torch.cuda.is_available():
        result.update(
            status="pending",
            reason="CUDA GPU unavailable; CPU interpreter timings are not performance data",
            correctness_smoke=cpu_smoke(),
        )
        return result
    result["status"] = "measured"
    result["gpu_name"] = torch.cuda.get_device_name()
    result["cases"] = [
        _gpu_case(batch, context, 16, 64, 16, warmup, iterations)
        for batch, context in [(1, 512), (4, 512), (4, 2048), (16, 2048)]
    ]
    return result
