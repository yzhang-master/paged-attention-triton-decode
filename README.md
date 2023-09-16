# paged-attention-triton-decode

> **Era:** 2023–2024 · PagedAttention technique from 2023, reimplemented in 2026 on Triton 3.x
> and PyTorch 2.x.

A fixed-capacity paged KV cache with per-request block tables and a native Triton decode kernel
that reads noncontiguous physical blocks. CPU execution uses Triton's interpreter for correctness,
never for performance claims.

## Problem

Contiguous KV storage makes variable-length request batches waste memory or require copying. Paged
storage fixes that, but an implementation that gathers KV into contiguous tensors before ordinary
attention loses the point of paging. Decode attention must follow each request's logical block
table and compute directly from physical pool storage.

## Method

The pool layout is `[physical_block, slot, head, head_dim]`, separately for keys and values. Each
request owns an ordered list of physical block IDs and a logical token length. Appends can
interleave requests, so consecutive logical blocks are generally noncontiguous.

The cache has a fixed block capacity. When space is needed, it evicts whole least-recently-used
requests, never partial sequences. A request larger than the entire pool fails before evicting
anyone. `release` explicitly returns a request's blocks. Block size must be a power of two.

For query `q`, each Triton program handles one `(request, head)` pair. It indirectly loads each
physical block from the table and uses online softmax across blocks:

```
m' = max(m, max(scores_in_block))
a  = a * exp(m - m') + Σ exp(score - m') * v
l  = l * exp(m - m') + Σ exp(score - m')
m  = m'
output = a / l
```

Scores are scaled by `1/sqrt(head_dim)`. Tail positions beyond the logical length and dimensions
beyond a non-power-of-two head size are masked. The Triton kernel performs no contiguous KV gather;
contiguous tensors appear only in the independent fp32 reference and benchmark baseline.

## Results

CPU correctness smoke over four requests with lengths **1, 5, 9, 14**, three heads, head dimension
12, and noncontiguous blocks:

| Check | Result |
|---|---:|
| maximum absolute error vs contiguous fp32 attention | **1.7881393e-7** |
| execution | Triton CPU interpreter |
| performance data | **no** |

The test suite also covers odd lengths, multiple requests, non-power-of-two head dimensions,
physical-block permutation, poisoned unused tail slots, large-score softmax stability, LRU
eviction, release, and oversize-request failure.

GPU latency against contiguous KV is **pending** because this machine has no CUDA GPU. No CPU
interpreter timing is presented as performance data.

| GPU | Batch | Context | Paged ms | Contiguous ms | Speedup |
|---|---:|---:|---:|---:|---:|
| _pending_ | 1, 4, 16 | 512, 2048 | _pending_ | _pending_ | _pending_ |

The measured-status JSON from the local run, including its pending reason and correctness smoke,
is in `results/gpu_latency.json`.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu  # use a CUDA build on GPU
pip install -e ".[dev]"
ruff check .
pytest
python scripts/bench_decode.py --smoke   # CPU correctness only
python scripts/bench_decode.py           # pending JSON on CPU; real CUDA-event timings on GPU
```

Without installing the package, prefix Python commands with `PYTHONPATH=src`. On a CUDA machine,
`scripts/bench_decode.py --warmup 10 --iterations 50` checks each output against contiguous
attention before timing it and writes `results/gpu_latency.json`.

## Limitations

- Decode only: one query token per request per launch; no paged prefill kernel.
- FP32 multi-head attention only; no GQA/MQA, quantization, or split-K decode.
- Fixed-capacity eviction removes whole requests and is a policy choice, not prefix sharing.
- Block-table creation and LRU timestamp updates are included in each public decode call.
- The CPU interpreter is a correctness tool and is orders of magnitude slower than native GPU
  execution. GPU performance remains unmeasured.

## References

- W. Kwon et al. *Efficient Memory Management for Large Language Model Serving with
  PagedAttention.* SOSP 2023. arXiv:2309.06180
- T. Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.*
  2022. arXiv:2205.14135 (online-softmax background)
- P. Tillet, H. T. Kung, D. Cox. *Triton: an Intermediate Language and Compiler for Tiled Neural
  Network Computations.* MAPL 2019

## License

MIT
