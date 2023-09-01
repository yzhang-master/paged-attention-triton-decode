import pytest
import torch

from paged_attention.cache import CacheCapacityError, PagedKVCache


def test_interleaved_requests_have_independent_noncontiguous_tables() -> None:
    cache = PagedKVCache(num_blocks=8, block_size=2, num_heads=2, head_dim=4)
    for request_id in (10, 20):
        cache.create_sequence(request_id)
    cache.append(10, torch.ones(3, 2, 4), torch.full((3, 2, 4), 10.0))
    cache.append(20, torch.ones(5, 2, 4), torch.full((5, 2, 4), 20.0))
    cache.append(10, torch.ones(2, 2, 4), torch.full((2, 2, 4), 10.0))
    table, lengths = cache.block_table([10, 20])
    assert table.tolist() == [[7, 6, 2], [5, 4, 3]]
    assert lengths.tolist() == [5, 5]
    assert len(set(table.flatten().tolist())) == 6


def test_lru_eviction_preserves_recent_request() -> None:
    cache = PagedKVCache(num_blocks=3, block_size=2, num_heads=1, head_dim=4)
    for request_id in (1, 2, 3):
        cache.create_sequence(request_id)
    cache.append(1, torch.randn(2, 1, 4), torch.randn(2, 1, 4))
    cache.append(2, torch.randn(2, 1, 4), torch.randn(2, 1, 4))
    cache.append(3, torch.randn(3, 1, 4), torch.randn(3, 1, 4))
    assert 1 not in cache.sequences
    assert set(cache.sequences) == {2, 3}
    assert cache.evictions == 1
    assert not cache.free_blocks


def test_capacity_failure_does_not_evict_for_oversized_request() -> None:
    cache = PagedKVCache(num_blocks=2, block_size=2, num_heads=1, head_dim=4)
    cache.create_sequence(1)
    cache.create_sequence(2)
    cache.append(2, torch.randn(2, 1, 4), torch.randn(2, 1, 4))
    with pytest.raises(CacheCapacityError, match="entire KV pool"):
        cache.append(1, torch.randn(5, 1, 4), torch.randn(5, 1, 4))
    assert set(cache.sequences) == {1, 2}
    assert cache.sequences[1].length == 0


def test_release_returns_all_blocks() -> None:
    cache = PagedKVCache(num_blocks=3, block_size=2, num_heads=1, head_dim=4)
    cache.create_sequence(8)
    cache.append(8, torch.randn(4, 1, 4), torch.randn(4, 1, 4))
    cache.release(8)
    assert len(cache.free_blocks) == 3
    with pytest.raises(KeyError):
        cache.contiguous_kv(8)


def test_block_size_must_be_power_of_two() -> None:
    with pytest.raises(ValueError, match="power of two"):
        PagedKVCache(num_blocks=2, block_size=3, num_heads=1, head_dim=4)
