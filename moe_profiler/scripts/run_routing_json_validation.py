#!/usr/bin/env python3
"""Batch MoE a2a validation using routing matrices stored in JSON.

This variant records aggregate all-to-all timings per layer (no per-edge breakdown).
"""

import argparse
import json
import os
from typing import Dict, List

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig

from tools.moe_profiler.measure import _init_dist, measure_once
from tools.moe_profiler.routing_map import RoutingMap
from tools.moe_profiler.utils import bytes_for_combine, bytes_for_dispatch


def _build_routing_map_from_matrix(
    matrix: List[List[int]],
    topk: int,
    num_experts: int,
    counts_are_assignments: bool,
) -> RoutingMap:
    """Construct a RoutingMap whose per-rank indices match the matrix counts."""
    ep_ws = len(matrix)
    per_rank = {}
    num_local = num_experts // ep_ws
    if num_local * ep_ws != num_experts:
        raise ValueError(f"num_experts={num_experts} must be divisible by EP world size={ep_ws}")

    for src, row in enumerate(matrix):
        rows: List[List[int]] = []
        rr_counters = [0 for _ in range(ep_ws)]
        for dst, count in enumerate(row):
            if count == 0:
                continue
            if counts_are_assignments:
                if topk <= 0:
                    raise ValueError("topk must be positive when interpreting counts as assignments")
                if count % topk != 0:
                    raise ValueError(
                        f"Edge ({src}->{dst}) count={count} is not divisible by topk={topk}"
                    )
                num_tokens = count // topk
            else:
                num_tokens = count
            for _ in range(num_tokens):
                slots: List[int] = []
                for _slot in range(topk):
                    local_id = rr_counters[dst] % num_local
                    slots.append(dst * num_local + local_id)
                    rr_counters[dst] += 1
                rows.append(slots)
        if rows:
            per_rank[src] = torch.tensor(rows, dtype=torch.long)
        else:
            per_rank[src] = torch.zeros((0, topk), dtype=torch.long)
    return RoutingMap(per_rank, topk=topk, num_experts=num_experts)


def _avg(values: List[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _run_layer(
    layer_name: str,
    matrix: List[List[int]],
    transformer_cfg: TransformerConfig,
    hidden_size: int,
    bh: int,
    bp: int,
    warmup: int,
    iters: int,
    seed_base: int,
    counts_are_assignments: bool,
) -> Dict:
    ep_ws = parallel_state.get_expert_model_parallel_world_size()
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    num_experts = transformer_cfg.num_moe_experts
    rm = _build_routing_map_from_matrix(
        matrix,
        transformer_cfg.moe_router_topk,
        num_experts,
        counts_are_assignments,
    )

    tokens_out = sum(int(v) for v in matrix[ep_rank]) if matrix else 0
    assignments_out = tokens_out if counts_are_assignments else tokens_out * transformer_cfg.moe_router_topk
    dispatch_bytes = bytes_for_dispatch(assignments_out, hidden_size, bh, bp)
    combine_bytes = bytes_for_combine(assignments_out, hidden_size, bh)

    for w in range(max(0, warmup)):
        _ = measure_once(transformer_cfg, rm, hidden_size, bh, bp, rng_seed=seed_base + 997 * (w + 1))

    total_ms: List[float] = []

    for step in range(max(1, iters)):
        result = measure_once(
            transformer_cfg,
            rm,
            hidden_size,
            bh,
            bp,
            rng_seed=seed_base + 1231 * (step + 1),
        )
        total = float(getattr(result, "total_cuda_ms", 0.0))
        if total <= 0.0:
            pre_local = float(result.preprocess_ms_per_rank.get(ep_rank, 0.0))
            total = pre_local + float(result.dispatch_total_ms) + float(result.combine_total_ms)
        total_ms.append(total)

    summary = {
        "layer": layer_name,
        "src_rank": ep_rank,
        "total_ms": _avg(total_ms),
        "tokens_out": int(tokens_out),
        "assignments_out": int(assignments_out),
        "dispatch_bytes": int(dispatch_bytes),
        "combine_bytes": int(combine_bytes),
        "total_bytes": int(dispatch_bytes + combine_bytes),
    }
    return summary


def _aggregate_layer(ep_ws: int, gathered: List[Dict]) -> Dict:
    total_ms_per_rank = [0.0 for _ in range(ep_ws)]
    dispatch_bytes_per_rank = [0 for _ in range(ep_ws)]
    combine_bytes_per_rank = [0 for _ in range(ep_ws)]
    total_bytes_per_rank = [0 for _ in range(ep_ws)]
    tokens_out_per_rank = [0 for _ in range(ep_ws)]
    assignments_per_rank = [0 for _ in range(ep_ws)]

    for item in gathered[:ep_ws]:
        if item is None:
            continue
        src = int(item["src_rank"])
        total_ms_per_rank[src] = float(item["total_ms"])
        dispatch_bytes_per_rank[src] = int(item["dispatch_bytes"])
        combine_bytes_per_rank[src] = int(item["combine_bytes"])
        total_bytes_per_rank[src] = int(item.get("total_bytes", dispatch_bytes_per_rank[src] + combine_bytes_per_rank[src]))
        tokens_out_per_rank[src] = int(item["tokens_out"])
        assignments_per_rank[src] = int(item["assignments_out"])

    total_avg = _avg(total_ms_per_rank)

    return {
        "total_ms_avg": total_avg,
        "total_ms_max": max(total_ms_per_rank) if total_ms_per_rank else 0.0,
        "total_ms_per_rank": total_ms_per_rank,
        "dispatch_bytes_per_rank": dispatch_bytes_per_rank,
        "combine_bytes_per_rank": combine_bytes_per_rank,
        "total_bytes_per_rank": total_bytes_per_rank,
        "tokens_out_per_rank": tokens_out_per_rank,
        "assignments_per_rank": assignments_per_rank,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute aggregate all-to-all timings from routing JSON.")
    parser.add_argument("--routing-json", required=True, help="Path to routing_*.json(l) file")
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--num-experts", type=int, required=True)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--bh", type=int, default=2, help="Bytes per hidden element")
    parser.add_argument("--bp", type=int, default=2, help="Bytes per probability element")
    parser.add_argument("--ep-size", type=int, help="Expert parallel size; inferred from matrix if omitted")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--counts-are-assignments",
        action="store_true",
        help="Treat matrix entries as token assignments (tokens * topk)",
    )
    parser.add_argument("--output", help="Optional path to write aggregated JSON results")
    args = parser.parse_args()

    with open(args.routing_json, "r") as f:
        routing_payload = json.load(f)

    layers = sorted(routing_payload.items(), key=lambda kv: kv[0])
    if not layers:
        raise ValueError("Routing JSON is empty")

    inferred_ep = len(layers[0][1])
    if args.ep_size and args.ep_size != inferred_ep:
        raise ValueError(
            f"Provided ep_size={args.ep_size} does not match matrix dimension={inferred_ep}"
        )
    ep_size = args.ep_size or inferred_ep

    _init_dist(tp=args.tp_size, pp=args.pp_size, ep=ep_size)
    ep_ws = parallel_state.get_expert_model_parallel_world_size()
    if ep_ws != ep_size:
        raise RuntimeError(f"Runtime EP world size {ep_ws} != configured {ep_size}")

    transformer_cfg = TransformerConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=max(1, args.hidden_size // args.hidden_size),
        num_layers=1,
        add_bias_linear=False,
        num_moe_experts=args.num_experts,
        moe_router_topk=args.topk,
        moe_token_dispatcher_type="alltoall",
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        expert_tensor_parallel_size=1,
    )

    ep_rank = parallel_state.get_expert_model_parallel_rank()
    world_rank = dist.get_rank()
    results = {}

    for idx, (layer_name, matrix) in enumerate(layers):
        if len(matrix) != ep_ws:
            raise ValueError(f"Layer {layer_name} matrix dimension {len(matrix)} != EP world size {ep_ws}")
        seed_base = args.seed + idx * 10007 + ep_rank * 257
        local_summary = _run_layer(
            layer_name,
            matrix,
            transformer_cfg,
            hidden_size=args.hidden_size,
            bh=args.bh,
            bp=args.bp,
            warmup=args.warmup_iters,
            iters=args.iters,
            seed_base=seed_base,
            counts_are_assignments=args.counts_are_assignments,
        )
        gather_list: List[Dict] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gather_list, local_summary)
        if world_rank == 0:
            layer_result = _aggregate_layer(ep_ws, gather_list[:ep_ws])
            results[layer_name] = layer_result
        dist.barrier()

    if world_rank == 0:
        output = {
            "routing_json": os.path.abspath(args.routing_json),
            "hidden_size": args.hidden_size,
            "num_experts": args.num_experts,
            "topk": args.topk,
            "bh": args.bh,
            "bp": args.bp,
            "warmup_iters": args.warmup_iters,
            "iters": args.iters,
            "counts_are_assignments": args.counts_are_assignments,
            "layers": results,
        }
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w") as wf:
                json.dump(output, wf, indent=2)
        else:
            print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
