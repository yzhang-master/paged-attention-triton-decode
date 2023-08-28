from __future__ import annotations

import math

import torch


def contiguous_attention(
    query: torch.Tensor, keys: list[torch.Tensor], values: list[torch.Tensor]
) -> torch.Tensor:
    """Fp32 attention over a separately contiguous KV tensor for each request."""
    if query.ndim != 3 or len(keys) != query.shape[0] or len(values) != len(keys):
        raise ValueError("query must be [batch, heads, dim] with one KV pair per request")
    outputs = []
    for batch, (key, value) in enumerate(zip(keys, values, strict=True)):
        if key.shape[0] == 0 or key.shape != value.shape:
            raise ValueError("each request needs nonempty, equally shaped keys and values")
        scores = torch.einsum("hd,lhd->hl", query[batch], key) / math.sqrt(key.shape[-1])
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hl,lhd->hd", probabilities, value))
    return torch.stack(outputs).float()
