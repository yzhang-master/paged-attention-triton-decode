from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from paged_attention.bench import benchmark, cpu_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="CPU correctness only; no timings")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("results/gpu_latency.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = cpu_smoke() if args.smoke else benchmark(args.warmup, args.iterations)
    if not args.smoke:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.output}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
