\
from __future__ import annotations
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch

@dataclass
class RoutingMap:
    """
    Routing map for all EP ranks.
    For each src EP rank r, store a (tokens, topk) int64 tensor of global expert ids (0..num_experts-1).
    """
    per_rank_indices: Dict[int, torch.Tensor]  # {src_rank: LongTensor [num_tokens, topk]}
    topk: int
    num_experts: int

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def build_random_routing_map(ep_world_size: int, num_experts: int, topk: int,
                             tokens_per_rank: int, seed: int) -> RoutingMap:
    per_rank: Dict[int, torch.Tensor] = {}
    gen = _rng(seed)
    for r in range(ep_world_size):
        # uniform select experts for each topk slot
        indices = gen.integers(low=0, high=num_experts, size=(tokens_per_rank, topk), dtype=np.int64)
        per_rank[r] = torch.from_numpy(indices.astype(np.int64))
    return RoutingMap(per_rank, topk=topk, num_experts=num_experts)

def build_random_routing_map_per_rank_tokens(ep_world_size: int, num_experts: int, topk: int,
                                             tokens_per_rank_per_rank: List[int], seed: int) -> RoutingMap:
    """Like build_random_routing_map but allows different token counts per source rank."""
    assert len(tokens_per_rank_per_rank) == ep_world_size
    per_rank: Dict[int, torch.Tensor] = {}
    gen = _rng(seed)
    for r in range(ep_world_size):
        tpr = int(max(0, tokens_per_rank_per_rank[r]))
        if tpr == 0:
            per_rank[r] = torch.zeros((0, topk), dtype=torch.long)
            continue
        indices = gen.integers(low=0, high=num_experts, size=(tpr, topk), dtype=np.int64)
        per_rank[r] = torch.from_numpy(indices.astype(np.int64))
    return RoutingMap(per_rank, topk=topk, num_experts=num_experts)

def build_uniform_routing_map(ep_world_size: int, num_experts: int, topk: int,
                              tokens_per_rank: int, exclude_self: bool = True) -> RoutingMap:
    """Build a routing map where each source rank sends equal number of assignments
    (token copies including topk) to every destination rank (optionally excluding self).

    Uses negative placeholder ids -(dst+1) which will be materialized later into
    concrete expert ids on the destination rank.
    """
    per_rank: Dict[int, torch.Tensor] = {}
    for src in range(ep_world_size):
        # Number of target destinations for this source
        dests = [d for d in range(ep_world_size) if (d != src) or not exclude_self]
        if len(dests) == 0 or tokens_per_rank <= 0:
            per_rank[src] = torch.zeros((0, topk), dtype=torch.long)
            continue
        total_assign = int(tokens_per_rank) * int(topk)
        nD = len(dests)
        base = total_assign // nD
        rem = total_assign % nD
        # Build a flat list with uniform counts per destination (first rem get +1)
        flat: List[int] = []
        for i, d in enumerate(dests):
            cnt = base + (1 if i < rem else 0)
            flat.extend([-(d + 1)] * cnt)
        # Safety: pad/truncate to exact length
        if len(flat) < total_assign:
            flat.extend([-(dests[-1] + 1)] * (total_assign - len(flat)))
        elif len(flat) > total_assign:
            flat = flat[: total_assign]
        arr = torch.tensor(flat, dtype=torch.long).view(tokens_per_rank, topk)
        per_rank[src] = arr
    return RoutingMap(per_rank, topk=topk, num_experts=num_experts)

def build_biased_routing_map_per_rank_tokens(ep_world_size: int, num_experts: int, num_local_experts: int,
                                             topk: int,
                                             tokens_per_rank_per_rank: List[int], seed: int,
                                             bias_src_rank: int, bias_dst_rank: int,
                                             bias_prob: float = 0.7) -> RoutingMap:
    """Like build_random_routing_map_per_rank_tokens but bias src->dst assignments.

    For the given bias_src_rank, each token's topk slots are independently chosen:
      - with probability bias_prob, pick an expert from bias_dst_rank's local experts;
      - otherwise pick uniformly from all experts.
    Other ranks remain uniform.
    """
    assert len(tokens_per_rank_per_rank) == ep_world_size
    per_rank: Dict[int, torch.Tensor] = {}
    gen = _rng(seed)
    # Precompute dst expert id range
    dst_start = bias_dst_rank * num_local_experts
    dst_end = dst_start + num_local_experts
    for r in range(ep_world_size):
        tpr = int(max(0, tokens_per_rank_per_rank[r]))
        if tpr == 0:
            per_rank[r] = torch.zeros((0, topk), dtype=torch.long)
            continue
        if r != bias_src_rank:
            indices = gen.integers(low=0, high=num_experts, size=(tpr, topk), dtype=np.int64)
            per_rank[r] = torch.from_numpy(indices.astype(np.int64))
        else:
            # Biased selection for src rank
            arr = np.empty((tpr, topk), dtype=np.int64)
            for i in range(tpr):
                for s in range(topk):
                    if gen.random() < float(bias_prob):
                        arr[i, s] = gen.integers(low=dst_start, high=dst_end, dtype=np.int64)
                    else:
                        arr[i, s] = gen.integers(low=0, high=num_experts, dtype=np.int64)
            per_rank[r] = torch.from_numpy(arr)
    return RoutingMap(per_rank, topk=topk, num_experts=num_experts)

def build_custom_routing_map_from_csv(csv_path: str, ep_world_size: int,
                                      num_experts: int, topk: int) -> RoutingMap:
    """
    CSV schema: src_rank,dst_ep_rank,num_tokens[,topk_slot]
    If topk>1 and topk_slot omitted, tokens are split evenly across topk slots towards dst rank's local experts.
    """
    # accumulate counts per (src, dst, slot) for reproducible construction
    counts: Dict[Tuple[int,int,int], int] = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        required = {"src_rank", "dst_ep_rank", "num_tokens"}
        if not required.issubset(set(cols)):
            raise ValueError(f"CSV must contain headers: {required}, but got {cols}")
        for row in reader:
            src = int(row["src_rank"])
            dst = int(row["dst_ep_rank"])
            n = int(row["num_tokens"])
            slot = int(row.get("topk_slot", -1))
            if slot == -1:
                # split evenly across slots
                base = n // topk
                rem = n % topk
                for s in range(topk):
                    extra = 1 if s < rem else 0
                    counts[(src, dst, s)] = counts.get((src, dst, s), 0) + base + extra
            else:
                if slot < 0 or slot >= topk:
                    raise ValueError(f"Invalid topk_slot={slot}, must be in [0,{topk-1}]")
                counts[(src, dst, slot)] = counts.get((src, dst, slot), 0) + n
    # allocate per src rank token lists
    per_rank_lists: Dict[int, List[List[int]]] = {r: [] for r in range(ep_world_size)}
    # For each source rank, create rows where each row is a token with topk expert ids
    # To simplify, we group by dst and slot. For each count, create that many tokens assigned to dst's *local* experts
    # If multiple local experts on dst, spread tokens round-robin across them per slot.
    for (src, dst, slot), n in sorted(counts.items()):
        # we will fill tokens per combination one-by-one
        for _ in range(n):
            per_rank_lists[src].append([-1]*topk)  # placeholder
    # Now assign experts: we need to know local experts on dst rank.
    # We will fill after we know num_local_experts at runtime (derived in measure.py).
    # Here we just store dst rank and slot along with count; we encode temp negative ids as -(dst+1) to mark placeholders.
    # Later, measure code will translate -(dst+1) into actual expert ids for dst rank using num_local_experts and offsets.
    per_rank_indices: Dict[int, torch.Tensor] = {}
    for src in range(ep_world_size):
        rows = per_rank_lists[src]
        if not rows:
            per_rank_indices[src] = torch.zeros((0, topk), dtype=torch.long)
            continue
        # As placeholder, store - (dst + 1) in the appropriate slot for each token row added above.
        # We'll leave the other slots -1 for now; measure code will fill them.
        # To reconstruct which dst each token belongs to, we need the order above; we re-read the counts iteration.
        fill_rows = []
        for (s, d, slot), n in sorted(counts.items()):
            if s != src: 
                continue
            for _ in range(n):
                fill_rows.append((d, slot))
        assert len(fill_rows) == len(rows)
        # Build placeholder tensor
        arr = -torch.ones((len(rows), topk), dtype=torch.long)
        for i, (d, slot) in enumerate(fill_rows):
            arr[i, slot] = -(d + 1)  # negative marker
        per_rank_indices[src] = arr
    return RoutingMap(per_rank_indices, topk=topk, num_experts=num_experts)

def materialize_custom_placeholders(rm: RoutingMap, num_local_experts: int) -> RoutingMap:
    """Translate placeholder dst markers -(dst+1) into concrete expert ids:
       For each token and each slot s, if rm.per_rank_indices[r][i,s] < 0 encode destination EP rank;
       replace with a real expert id on that rank via round-robin among its local experts.
    """
    new_map: Dict[int, torch.Tensor] = {}
    rr_counters: Dict[int, int] = {}
    for src, idx in rm.per_rank_indices.items():
        if idx.numel() == 0:
            new_map[src] = idx.clone()
            continue
        out = idx.clone()
        for i in range(out.shape[0]):
            for s in range(out.shape[1]):
                v = int(out[i, s].item())
                if v < 0:
                    dst = -v - 1
                    # Round-robin assign local expert id
                    rr = rr_counters.get(dst, 0)
                    expert_global = dst * num_local_experts + (rr % num_local_experts)
                    rr_counters[dst] = rr + 1
                    out[i, s] = expert_global
        new_map[src] = out
    return RoutingMap(new_map, topk=rm.topk, num_experts=rm.num_experts)
