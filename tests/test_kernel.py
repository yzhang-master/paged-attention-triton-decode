import pytest
import torch

from paged_attention.cache import PagedKVCache
from paged_attention.kernel import paged_decode_attention
from paged_attention.reference import contiguous_attention


def make_cache(
    lengths: list[int], block_size: int, num_heads: int, head_dim: int
) -> tuple[PagedKVCache, list[int], torch.Tensor]:
    num_blocks = sum((length + block_size - 1) // block_size for length in lengths) + 2
    cache = PagedKVCache(num_blocks, block_size, num_heads, head_dim)
    request_ids = []
    generator = torch.Generator().manual_seed(41 + len(lengths) + head_dim)
    for request_id, length in enumerate(lengths, start=1):
        cache.create_sequence(request_id)
        keys = torch.randn(length, num_heads, head_dim, generator=generator)
        values = torch.randn(length, num_heads, head_dim, generator=generator)
        split = length // 2
        cache.append(request_id, keys[:split], values[:split])
        cache.append(request_id, keys[split:], values[split:])
        request_ids.append(request_id)
    query = torch.randn(len(lengths), num_heads, head_dim, generator=generator)
    return cache, request_ids, query


def reference_for(
    cache: PagedKVCache, request_ids: list[int], query: torch.Tensor
) -> torch.Tensor:
    pairs = [cache.contiguous_kv(request_id) for request_id in request_ids]
    return contiguous_attention(query, [pair[0] for pair in pairs], [pair[1] for pair in pairs])


@pytest.mark.parametrize(
    ("lengths", "block_size", "num_heads", "head_dim"),
    [
        ([1, 3, 5, 9], 4, 2, 8),
        ([2, 7, 8, 13], 2, 3, 12),
        ([1, 2, 17], 4, 1, 32),
    ],
)
def test_paged_kernel_matches_contiguous_fp32(
    lengths: list[int], block_size: int, num_heads: int, head_dim: int
) -> None:
    cache, request_ids, query = make_cache(lengths, block_size, num_heads, head_dim)
    actual = paged_decode_attention(query, cache, request_ids)
    expected = reference_for(cache, request_ids, query)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_physical_block_permutation_preserves_logical_attention() -> None:
    cache, request_ids, query = make_cache([11], 2, 2, 8)
    expected = paged_decode_attention(query, cache, request_ids)
    sequence = cache.sequences[request_ids[0]]
    old_blocks = sequence.blocks.copy()
    new_blocks = list(reversed(old_blocks))
    old_keys = cache.keys[old_blocks].clone()
    old_values = cache.values[old_blocks].clone()
    cache.keys[new_blocks] = old_keys
    cache.values[new_blocks] = old_values
    sequence.blocks = new_blocks
    actual = paged_decode_attention(query, cache, request_ids)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_unused_tail_slots_are_masked() -> None:
    cache, request_ids, query = make_cache([5, 7], 4, 2, 8)
    expected = paged_decode_attention(query, cache, request_ids)
    for request_id in request_ids:
        sequence = cache.sequences[request_id]
        first_unused = sequence.length % cache.block_size
        cache.keys[sequence.blocks[-1], first_unused:] = 1e6
        cache.values[sequence.blocks[-1], first_unused:] = -1e6
    actual = paged_decode_attention(query, cache, request_ids)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_online_softmax_handles_large_scores() -> None:
    cache, request_ids, query = make_cache([9], 4, 2, 8)
    cache.keys.mul_(20.0)
    query.mul_(20.0)
    actual = paged_decode_attention(query, cache, request_ids)
    expected = reference_for(cache, request_ids, query)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


def test_decode_rejects_empty_sequence() -> None:
    cache = PagedKVCache(2, 2, 1, 8)
    cache.create_sequence(1)
    with pytest.raises(ValueError, match="nonempty"):
        paged_decode_attention(torch.randn(1, 1, 8), cache, [1])
