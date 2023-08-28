from __future__ import annotations

import math

import torch

from paged_attention._triton import tl, triton
from paged_attention.cache import PagedKVCache


@triton.jit
def _paged_decode_kernel(
    query_ptr,
    keys_ptr,
    values_ptr,
    table_ptr,
    lengths_ptr,
    output_ptr,
    query_batch_stride: tl.constexpr,
    query_head_stride: tl.constexpr,
    table_batch_stride: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_dim: tl.constexpr,
    block_size: tl.constexpr,
    scale: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    offsets_d = tl.arange(0, block_dim)
    valid_d = offsets_d < head_dim
    query = tl.load(
        query_ptr + batch * query_batch_stride + head * query_head_stride + offsets_d,
        mask=valid_d,
        other=0.0,
    ).to(tl.float32)
    length = tl.load(lengths_ptr + batch)
    running_max = tl.full((), -float("inf"), tl.float32)
    running_sum = tl.zeros((), dtype=tl.float32)
    accumulator = tl.zeros((block_dim,), dtype=tl.float32)
    for logical_block in range(tl.cdiv(length, block_size)):
        physical_block = tl.load(table_ptr + batch * table_batch_stride + logical_block)
        offsets_n = tl.arange(0, block_size)
        positions = logical_block * block_size + offsets_n
        valid_n = positions < length
        cache_offsets = (
            physical_block * block_size * num_heads * head_dim
            + offsets_n[:, None] * num_heads * head_dim
            + offsets_d[None, :]
        )
        head_offset = head * head_dim
        key = tl.load(
            keys_ptr + head_offset + cache_offsets,
            mask=valid_n[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)
        value = tl.load(
            values_ptr + head_offset + cache_offsets,
            mask=valid_n[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(key * query[None, :], axis=1) * scale
        scores = tl.where(valid_n, scores, -float("inf"))
        next_max = tl.maximum(running_max, tl.max(scores, axis=0))
        correction = tl.exp(running_max - next_max)
        probabilities = tl.exp(scores - next_max)
        accumulator = accumulator * correction + tl.sum(
            probabilities[:, None] * value, axis=0
        )
        running_sum = running_sum * correction + tl.sum(probabilities, axis=0)
        running_max = next_max
    output = accumulator / running_sum
    tl.store(
        output_ptr + batch * query_batch_stride + head * query_head_stride + offsets_d,
        output,
        mask=valid_d,
    )


def paged_decode_attention(
    query: torch.Tensor, cache: PagedKVCache, request_ids: list[int]
) -> torch.Tensor:
    """Decode one token per request by indirectly reading physical KV blocks."""
    expected = (len(request_ids), cache.num_heads, cache.head_dim)
    if query.shape != expected or query.dtype != torch.float32:
        raise ValueError(f"query must be fp32 with shape {expected}")
    if query.device != cache.device:
        raise ValueError("query and cache must be on the same device")
    table, lengths = cache.block_table(request_ids)
    if torch.any(lengths <= 0):
        raise ValueError("decode attention requires nonempty sequences")
    output = torch.empty_like(query)
    grid = (len(request_ids), cache.num_heads)
    _paged_decode_kernel[grid](
        query,
        cache.keys,
        cache.values,
        table,
        lengths,
        output,
        query_batch_stride=query.stride(0),
        query_head_stride=query.stride(1),
        table_batch_stride=table.stride(0),
        num_heads=cache.num_heads,
        head_dim=cache.head_dim,
        block_dim=triton.next_power_of_2(cache.head_dim),
        block_size=cache.block_size,
        scale=1.0 / math.sqrt(cache.head_dim),
        num_warps=4,
    )
    return output
