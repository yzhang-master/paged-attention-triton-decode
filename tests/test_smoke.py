import torch

from paged_attention import PagedKVCache


def test_paged_cache_read_write_smoke() -> None:
    cache = PagedKVCache(num_blocks=3, block_size=2, num_heads=2, head_dim=4)
    cache.create_sequence(7)
    keys = torch.randn(3, 2, 4)
    values = torch.randn(3, 2, 4)
    cache.append(7, keys, values)
    table, lengths = cache.block_table([7])
    assert table.tolist() == [[2, 1]]
    assert lengths.tolist() == [3]
    actual_k, actual_v = cache.contiguous_kv(7)
    torch.testing.assert_close(actual_k, keys)
    torch.testing.assert_close(actual_v, values)
