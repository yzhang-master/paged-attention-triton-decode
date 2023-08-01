# paged-attention-triton-decode

> **Era:** 2023–2024 · PagedAttention technique from 2023, reimplemented in 2026 on Triton 3.x.

A fixed-capacity paged KV cache and decode attention over physical blocks selected by a block
table. CPU execution uses Triton's interpreter for correctness, never for performance claims.

## Status

The scaffold contains block allocation and a cache read/write smoke test. Eviction, the native
Triton kernel, correctness tests, and benchmarks are implemented in subsequent commits.

## Results

GPU latency against contiguous KV is **pending**. No CPU timing is presented as kernel
performance data.

## License

MIT
