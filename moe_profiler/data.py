\
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import torch

def derive_hidden_and_probs_for_rank(
    routing_indices: torch.Tensor, hidden_size: int, dtype: torch.dtype, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given routing_indices [num_tokens, topk], generate:
      hidden_states: [num_tokens, hidden_size] on current CUDA device with requested dtype
      probs:         [num_tokens, topk] with rows summing to 1.0 (same dtype as hidden or float32)
    """
    device = torch.cuda.current_device()
    num_tokens, topk = routing_indices.shape
    # Hidden states
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    hidden_states = torch.randn((num_tokens, hidden_size), device=device, dtype=dtype, generator=gen)
    # Probs: sample from Dirichlet or uniform if topk==1
    if topk == 1:
        probs = torch.ones((num_tokens, 1), device=device, dtype=torch.float32)
    else:
        # Gamma sampling for Dirichlet
        alpha = torch.ones((num_tokens, topk), device=device, dtype=torch.float32)
        gamma = torch.distributions.Gamma(alpha, torch.ones_like(alpha)).sample()
        probs = gamma / (gamma.sum(dim=-1, keepdim=True) + 1e-8)
    return hidden_states, probs
