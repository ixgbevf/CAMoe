\
from __future__ import annotations
import os, sys, json, math, time, inspect, random
from collections import defaultdict
from typing import Dict, Tuple, List, Any, Optional
import torch
import torch.distributed as dist

# Support both package and script execution
try:
    from .io_utils import load_config, dump_json
    from .config import from_dict, ProfilerConfig
    from .routing_map import (
        build_random_routing_map, build_random_routing_map_per_rank_tokens, build_biased_routing_map_per_rank_tokens,
        build_uniform_routing_map,
        build_custom_routing_map_from_csv, materialize_custom_placeholders, RoutingMap
    )
    from .measure import _init_dist, measure_once
    from .fit import grid_fit_per_edge
    from .visualize import plot_fit_svg
    from .utils import bytes_for_dispatch, bytes_for_combine
except ImportError:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from tools.moe_profiler.io_utils import load_config, dump_json
    from tools.moe_profiler.config import from_dict, ProfilerConfig
    from tools.moe_profiler.routing_map import (
        build_random_routing_map, build_random_routing_map_per_rank_tokens, build_biased_routing_map_per_rank_tokens,
        build_uniform_routing_map,
        build_custom_routing_map_from_csv, materialize_custom_placeholders, RoutingMap
    )
    from tools.moe_profiler.measure import _init_dist, measure_once
    from tools.moe_profiler.fit import grid_fit_per_edge
    from tools.moe_profiler.visualize import plot_fit_svg
    from tools.moe_profiler.utils import bytes_for_dispatch, bytes_for_combine

def _load_fit_if_exists(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _predict_edge_ms(alpha_ms, beta_ms_per_byte, i: int, j: int, bytes_: int) -> float:
    try:
        a = alpha_ms[i][j]
        b = beta_ms_per_byte[i][j]
        # If coefficients look missing/unreliable (both zero/NaN/inf), fallback to bytes heuristic
        import math as _m
        if not _m.isfinite(a) or not _m.isfinite(b) or (abs(a) < 1e-12 and abs(b) < 1e-18):
            return float(bytes_)
        return float(a) + float(b) * float(bytes_)
    except Exception:
        # Fall back to proportional-bytes heuristic
        return float(bytes_)

def _build_baseline_map_for_edge(ep_ws: int, num_experts: int, topk: int, src_rank: int, dst_rank: int,
                                 M_tokens: int, num_local_experts: int) -> RoutingMap:
    import torch as _torch
    per_rank = {}
    for r in range(ep_ws):
        if r != src_rank:
            per_rank[r] = _torch.zeros((0, topk), dtype=_torch.long)
            continue
        arr = _torch.empty((M_tokens, topk), dtype=_torch.long)
        # Round-robin pick local experts on dst for each slot
        for t in range(M_tokens):
            for s in range(topk):
                expert_local = (t + s) % num_local_experts
                arr[t, s] = dst_rank * num_local_experts + expert_local
        per_rank[r] = arr
    return RoutingMap(per_rank, topk=topk, num_experts=num_experts)

def _select_random_map_for_rank(ep_ws: int, num_experts: int, topk: int, tokens_per_rank: int, seed: int, rank: int) -> RoutingMap:
    # Each rank builds its own random routing; all ranks should use the same tokens_per_rank to emulate congestion.
    return build_random_routing_map(ep_world_size=ep_ws, num_experts=num_experts, topk=topk,
                                    tokens_per_rank=tokens_per_rank, seed=seed + rank)

def _edge_bytes_from_map(rm: RoutingMap, hidden_size: int, bh: int, bp: int, num_local_experts: int, src_rank: int) -> Dict[int, Tuple[int, int]]:
    """Return {dst_rank: (Bdisp_ij, Bcomb_ij)} for this source rank"""
    import torch as _torch
    idx = rm.per_rank_indices.get(src_rank, _torch.zeros((0, rm.topk), dtype=_torch.long))
    if idx.numel() == 0:
        return {}
    dst_ranks = (idx // num_local_experts).cpu().numpy().reshape(-1)
    counts = defaultdict(int)
    for j in dst_ranks.tolist():
        counts[int(j)] += 1
    out = {}
    for j, M in counts.items():
        out[j] = (bytes_for_dispatch(M, hidden_size, bh, bp), bytes_for_combine(M, hidden_size, bh))
    return out

def _make_balanced_tokens_per_rank_list(ep_ws: int, base_tokens: int, jitter_ratio: float, seed: int) -> List[int]:
    import random as _pyr
    _pyr.seed(int(seed))
    # Interpret jitter_ratio as the maximum allowed range ratio across ranks.
    # Constrain each rank within +/- (jitter_ratio/2) so that max-min <= jitter_ratio.
    half_ratio = max(0.0, float(jitter_ratio)) / 2.0
    delta = max(1, int(round(base_tokens * half_ratio)))
    vals = [base_tokens + _pyr.randint(-delta, delta) for _ in range(ep_ws)]
    total_target = ep_ws * base_tokens
    diff = sum(vals) - total_target
    if diff != 0:
        step = 1 if diff > 0 else -1
        k = 0
        lo, hi = base_tokens - delta, base_tokens + delta
        while diff != 0 and k < ep_ws * (2 * delta + 4):
            i = k % ep_ws
            newv = vals[i] - step
            if lo <= newv <= hi and newv >= 0:
                vals[i] = newv
                diff -= step
            k += 1
    lo, hi = base_tokens - delta, base_tokens + delta
    vals = [min(max(v, max(0, lo)), hi) for v in vals]
    # Enforce final range bound: max-min <= jitter_ratio * base_tokens (approx)
    if max(vals) - min(vals) > max(1, int(round(base_tokens * jitter_ratio))):
        # Center values to reduce spread
        mid = sum(vals) // ep_ws
        lo2, hi2 = base_tokens - delta, base_tokens + delta
        vals = [min(max(mid, lo2), hi2) for _ in range(ep_ws)]
    return vals

def _maybe_plot(cfg: ProfilerConfig, ep_ws: int, all_pre: Dict[int, List[float]], 
                all_disp: Dict[Tuple[int,int], Dict[str, List[Tuple[float,float]]]],
                all_comb: Dict[Tuple[int,int], Dict[str, List[Tuple[float,float]]]],
                alpha_disp, beta_disp, alpha_comb, beta_comb):
    if not cfg.plots.enable:
        return
    # Plots for edges
    for (i,j), d in all_disp.items():
        xs = [b for (b, ms) in d["dispatch"]]
        ys = [ms for (b, ms) in d["dispatch"]]
        a = alpha_disp[i][j]
        b = beta_disp[i][j]
        title = f"Dispatch edge {i}->{j}"
        out = os.path.join(cfg.paths.plots_dir, f"disp_edge_{i}{j}.svg")
        try:
            plot_fit_svg(xs, ys, a, b, title, out)
        except Exception:
            pass
    for (i,j), d in all_comb.items():
        xs = [b for (b, ms) in d["combine"]]
        ys = [ms for (b, ms) in d["combine"]]
        a = alpha_comb[i][j]
        b = beta_comb[i][j]
        title = f"Combine edge {i}->{j}"
        out = os.path.join(cfg.paths.plots_dir, f"comb_edge_{i}{j}.svg")
        try:
            plot_fit_svg(xs, ys, a, b, title, out)
        except Exception:
            pass
    # Preprocess by rank
    try:
        import matplotlib.pyplot as plt
        ranks = list(range(ep_ws))
        vals = [sum(all_pre[r])/max(1,len(all_pre[r])) if r in all_pre else 0.0 for r in ranks]
        plt.figure()
        plt.scatter(ranks, vals, label="preprocess(ms)")
        plt.plot(ranks, vals, label="avg")
        plt.xlabel("EP rank")
        plt.ylabel("Latency (ms)")
        # Omit title per request
        out = os.path.join(cfg.paths.plots_dir, f"preprocess_by_rank.svg")
        plt.legend()
        plt.savefig(out, format="svg")
        plt.close()
    except Exception:
        pass

def main(argv: List[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 1:
        print("Usage: python -m moe_profiler.main <config.yaml|json>")
        sys.exit(1)
    cfg_path = argv[0]
    cfg_dict = load_config(cfg_path)
    cfg = from_dict(cfg_dict)
    os.makedirs(cfg.paths.data_dir, exist_ok=True)
    os.makedirs(cfg.paths.measurements_dir, exist_ok=True)
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    os.makedirs(cfg.paths.plots_dir, exist_ok=True)

    # Init distributed and parallel state
    _init_dist(tp=cfg.parallel.tp_size, pp=cfg.parallel.pp_size, ep=cfg.parallel.ep_size)

    # Create a synchronized run_id (timestamp) across all ranks
    run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime()) if dist.get_rank() == 0 else ""
    obj = [run_id]
    try:
        dist.broadcast_object_list(obj, src=0)
    except Exception:
        # Fallback: assume ranks start within the same second
        pass
    run_id = obj[0] if obj[0] else time.strftime("%Y%m%d_%H%M%S", time.localtime())

    # Timestamped sub-directories for this run
    meas_dir_ts = os.path.join(cfg.paths.measurements_dir, run_id)
    results_dir_ts = os.path.join(cfg.paths.results_dir, run_id)
    plots_dir_ts = os.path.join(cfg.paths.plots_dir, run_id)
    os.makedirs(meas_dir_ts, exist_ok=True)
    os.makedirs(results_dir_ts, exist_ok=True)
    os.makedirs(plots_dir_ts, exist_ok=True)

    # EP dimensions
    from megatron.core import parallel_state
    ep_ws = parallel_state.get_expert_model_parallel_world_size()
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    ep_group = parallel_state.get_expert_model_parallel_group()
    num_local = cfg.model.num_experts // ep_ws

    # Build transformer config
    from megatron.core.transformer.transformer_config import TransformerConfig
    transformer_cfg = TransformerConfig(
        # Minimal required transformer attributes
        hidden_size=cfg.model.hidden_size,
        num_attention_heads=max(1, int(cfg.model.hidden_size // cfg.model.hidden_size)),  # at least 1
        num_layers=1,
        add_bias_linear=False,

        # MoE-specific settings
        num_moe_experts=cfg.model.num_experts,
        moe_router_topk=cfg.model.topk,
        moe_token_dispatcher_type="alltoall",

        # Parallelism
        tensor_model_parallel_size=cfg.parallel.tp_size,
        pipeline_model_parallel_size=cfg.parallel.pp_size,
        expert_tensor_parallel_size=1,
    )

    # Bytes per element
    bh = cfg.model.bh
    bp = cfg.model.bp

    # Aggregate containers
    preprocess_lat_ms: Dict[int, List[float]] = defaultdict(list)
    disp_meas: Dict[Tuple[int,int], List[Tuple[float, float]]] = defaultdict(list)
    comb_meas: Dict[Tuple[int,int], List[Tuple[float, float]]] = defaultdict(list)

    # Scenario routing logic
    scenario = cfg.scene.scenario
    # Optional container for coarse-baseline per-edge total (pre+disp+comb) pairs on this rank
    local_coarse_total_ms_per_edge: Dict[Tuple[int,int], List[Tuple[float, float]]] = defaultdict(list)
    # Generic local timing records container (exists for all scenarios)
    local_records: List[dict] = []

    if scenario == "baseline":
        # Coordinate all ranks to measure exactly one (src, dst) edge at a time.
        # Supports dispatch-side selection: sender (default), receiver, or both.
        warmup = int(getattr(cfg.measure, 'warmup_iters', 0))
        iters = int(getattr(cfg.measure, 'iters', 1))

        def _targets_for_src(i: int) -> List[int]:
            if cfg.measure.baseline_target_dst is not None:
                return [int(cfg.measure.baseline_target_dst)]
            if getattr(cfg.measure, 'baseline_only_lower_triangle', False):
                return [j for j in range(ep_ws) if i > j]
            else:
                return [j for j in range(ep_ws) if j != i]

        # Build global list of pairs on rank 0, then broadcast to all ranks.
        pairs: List[Tuple[int, int]] = []
        if dist.get_rank() == 0:
            for i in range(ep_ws):
                for j in _targets_for_src(i):
                    pairs.append((i, j))
        obj = [pairs]
        try:
            dist.broadcast_object_list(obj, src=0)
            pairs = obj[0]
        except Exception:
            # Fallback: derive locally; all ranks will derive the same list
            pairs = [(i, j) for i in range(ep_ws) for j in _targets_for_src(i)]

        # Iterate sizes, then each pair sequentially (globally synchronized).
        local_records: List[dict] = []
        for M in cfg.measure.sizes:
            for (src, dst) in pairs:
                # Align all ranks before configuring this pair
                try:
                    dist.barrier()
                except Exception:
                    pass
                rm_edge = _build_baseline_map_for_edge(ep_ws, cfg.model.num_experts, cfg.model.topk,
                                                       src_rank=src, dst_rank=dst, M_tokens=M,
                                                       num_local_experts=num_local)
                seed_base = cfg.routing.random.seed + M + src * 131 + dst * 17
                # Warmup
                for w in range(max(0, warmup)):
                    _ = measure_once(transformer_cfg, rm_edge, cfg.model.hidden_size, bh, bp,
                                      rng_seed=seed_base + 997 * (w + 1))
                # Measure and average
                pre_sums: Dict[int, float] = defaultdict(float)
                disp_sum_sender = 0.0
                disp_sum_receiver = 0.0
                comb_sum_receiver = 0.0
                # CUDA event local timings (sender-side diff)
                disp_cuda_sum_sender = 0.0
                total_cuda_sum_sender = 0.0
                # Local timers (this rank)
                local_disp_sum = 0.0
                local_comb_sum = 0.0
                cnt = 0
                for t_idx in range(max(1, iters)):
                    t = measure_once(transformer_cfg, rm_edge, cfg.model.hidden_size, bh, bp,
                                     rng_seed=seed_base + 1231 * (t_idx + 1))
                    for r, ms in t.preprocess_ms_per_rank.items():
                        pre_sums[r] += ms
                    # Sender-side dispatch time (host window)
                    if (src, dst) in t.dispatch_ms_per_edge and t.dispatch_ms_per_edge[(src, dst)]:
                        disp_sum_sender += t.dispatch_ms_per_edge[(src, dst)][0][1]
                    # Sender-side CUDA event timings
                    if parallel_state.get_expert_model_parallel_rank() == src:
                        disp_cuda_sum_sender += float(getattr(t, 'dispatch_cuda_ms', 0.0))
                        total_cuda_sum_sender += float(getattr(t, 'total_cuda_ms', 0.0))
                    # Receiver-side dispatch time: on dst rank, use the rank-local total dispatch time window.
                    if parallel_state.get_expert_model_parallel_rank() == dst:
                        disp_sum_receiver += float(getattr(t, 'dispatch_total_ms', 0.0))
                    # Combine timing accumulation per mode
                    mode = str(getattr(cfg.measure, 'combine_timing_mode', 'receiver_local')).lower()
                    if mode == 'cross_rank_diff':
                        # Gather host timestamps from all EP ranks and compute cross-end diff
                        import torch as _torch
                        times_t = _torch.tensor(
                            [float(getattr(t, 'combine_host_t_start_ms', 0.0)), float(getattr(t, 'combine_host_t_end_ms', 0.0))],
                            dtype=_torch.float64,
                            device=_torch.cuda.current_device(),
                        )
                        gather_list = [ _torch.zeros_like(times_t) for _ in range(ep_ws) ]
                        try:
                            dist.all_gather(gather_list, times_t, group=ep_group)
                        except Exception:
                            # Fallback: use local timing as approximate
                            gather_list = [times_t for _ in range(ep_ws)]
                        # Direction selection
                        direction = str(getattr(cfg.measure, 'combine_diff_direction', 'sender_to_receiver')).lower()
                        try:
                            if direction == 'receiver_to_sender':
                                start_ms = float(gather_list[src][0].item())  # receiver start
                                end_ms   = float(gather_list[dst][1].item())  # sender end
                            else:
                                # default sender_to_receiver
                                start_ms = float(gather_list[dst][0].item())  # sender start
                                end_ms   = float(gather_list[src][1].item())  # receiver end
                            cross_ms = float(end_ms - start_ms)
                        except Exception:
                            cross_ms = float(getattr(t, 'combine_total_ms', 0.0))
                        # Accumulate cross diff once per-rank; we will attribute to dst below
                        comb_sum_receiver += cross_ms
                    elif mode == 'max_of_locals':
                        # Need both sides local windows; gather combine_total_ms across ranks
                        import torch as _torch
                        local_ms = _torch.tensor([float(getattr(t, 'combine_total_ms', 0.0))], dtype=_torch.float64, device=_torch.cuda.current_device())
                        gather_list = [ _torch.zeros_like(local_ms) for _ in range(ep_ws) ]
                        try:
                            dist.all_gather(gather_list, local_ms, group=ep_group)
                            ms_src = float(gather_list[src][0].item())
                            ms_dst = float(gather_list[dst][0].item())
                            comb_sum_receiver += max(ms_src, ms_dst)
                        except Exception:
                            comb_sum_receiver += float(getattr(t, 'combine_total_ms', 0.0))
                    else:
                        # receiver_local (legacy): only receiver accumulates its local window
                        if parallel_state.get_expert_model_parallel_rank() == dst:
                            comb_sum_receiver += float(getattr(t, 'combine_total_ms', 0.0))
                    # Local timers accumulate regardless of role
                    local_disp_sum += float(getattr(t, 'dispatch_total_ms', 0.0))
                    local_comb_sum += float(getattr(t, 'combine_total_ms', 0.0))
                    cnt += 1

                # Record averaged preprocess per rank
                for r, s in pre_sums.items():
                    preprocess_lat_ms[r].append(s / max(1, cnt))

                # Compute bytes for this edge (include topk duplication)
                M_assign = M * cfg.model.topk
                bdisp = bytes_for_dispatch(M_assign, cfg.model.hidden_size, bh, bp)
                bcomb = bytes_for_combine(M_assign, cfg.model.hidden_size, bh)

                # Attribute dispatch by side (but prefer CUDA ms on sender when available)
                side = str(getattr(cfg.measure, 'dispatch_side', 'sender')).lower()
                if side not in ("sender", "receiver", "both"):
                    side = "sender"
                if side in ("sender", "both") and ep_rank == src:
                    disp_time = float(disp_cuda_sum_sender / max(1, cnt)) if disp_cuda_sum_sender > 0 else float(disp_sum_sender / max(1, cnt))
                    disp_meas[(src, dst)].append((bdisp, disp_time))
                if side in ("receiver", "both") and ep_rank == dst:
                    disp_meas[(src, dst)].append((bdisp, disp_sum_receiver / max(1, cnt)))

                # Combine attribution
                if bool(getattr(cfg.measure, 'combine_from_sender_diff', False)):
                    if ep_rank == src:
                        avg_disp_cuda = float(disp_cuda_sum_sender / max(1, cnt))
                        avg_total_cuda = float(total_cuda_sum_sender / max(1, cnt))
                        comb_from_diff = max(0.0, avg_total_cuda - avg_disp_cuda)
                        comb_meas[(src, dst)].append((bcomb, comb_from_diff))
                else:
                    # Existing modes: receiver_local / cross_rank_diff / max_of_locals recorded on receiver
                    if ep_rank == dst:
                        comb_meas[(src, dst)].append((bcomb, comb_sum_receiver / max(1, cnt)))
                # Align after recording
                try:
                    dist.barrier()
                except Exception:
                    pass
                # Save per-rank local timers record for this (M, src, dst)
                local_records.append({
                    "M": int(M),
                    "src": int(src),
                    "dst": int(dst),
                    "dispatch_ms": float(local_disp_sum / max(1, cnt)),
                    "combine_ms": float(local_comb_sum / max(1, cnt)),
                })

    elif scenario == "all2all_profile":
        # Need previous fit (alpha/beta) to predict longest edge. If not available, use bytes heuristic.
        # Baseline fit can come from a separate directory or a direct file path
        baseline_dir_or_file = cfg.paths.baseline_results_dir or cfg.paths.results_dir
        if baseline_dir_or_file and baseline_dir_or_file.endswith('.json'):
            baseline_path = baseline_dir_or_file
        else:
            baseline_path = os.path.join(baseline_dir_or_file, "fit_results.json")
        baseline_fit = _load_fit_if_exists(baseline_path)
        alpha_disp = baseline_fit["dispatch"]["alpha_ms"] if baseline_fit else None
        beta_disp  = baseline_fit["dispatch"]["beta_ms_per_byte"] if baseline_fit else None

        warmup = int(getattr(cfg.measure, 'warmup_iters', 0))
        iters = int(getattr(cfg.measure, 'iters', 1))
        jitter = float(getattr(cfg.routing.random, 'jitter_ratio', 0.1))
        bias_prob = float(getattr(cfg.routing.random, 'bias_prob', 0.7))
        for M in cfg.measure.sizes:
            # Run a fixed number of attempts to keep collectives aligned across ranks.
            max_total_attempts = max(1, int(cfg.measure.max_attempts) * max(1, ep_ws))
            # Exclude self-edge by default; focus on inter-rank edges
            remaining_js = set(j for j in range(ep_ws) if j != ep_rank)
            for attempts in range(max_total_attempts):
                seed_cur = cfg.routing.random.seed + M + attempts * 17
                tokens_list = _make_balanced_tokens_per_rank_list(ep_ws, M, jitter, seed=seed_cur)
                # Choose a target edge to promote to longest for this src rank, if any remaining.
                target_j = min(remaining_js) if remaining_js else None
                if target_j is not None:
                    # Escalate bias over attempts to ensure coverage when baseline alpha/beta skew is large
                    bias_p_eff = max(0.5, min(0.99, bias_prob + 0.05 * attempts))
                    rm_try = build_biased_routing_map_per_rank_tokens(
                        ep_world_size=ep_ws,
                        num_experts=cfg.model.num_experts,
                        num_local_experts=num_local,
                        topk=cfg.model.topk,
                        tokens_per_rank_per_rank=tokens_list,
                        seed=seed_cur,
                        bias_src_rank=ep_rank,
                        bias_dst_rank=target_j,
                        bias_prob=bias_p_eff,
                    )
                else:
                    rm_try = build_random_routing_map_per_rank_tokens(
                        ep_world_size=ep_ws,
                        num_experts=cfg.model.num_experts,
                        topk=cfg.model.topk,
                        tokens_per_rank_per_rank=tokens_list,
                        seed=seed_cur,
                    )
                # Predict per-edge times and select the actual longest
                edge_bytes = _edge_bytes_from_map(rm_try, cfg.model.hidden_size, bh, bp, num_local, ep_rank)
                pred = {}
                for jd, (bdisp, _bcomb) in edge_bytes.items():
                    # Skip self-edge (i==j) when selecting the longest edge
                    if int(jd) == int(ep_rank):
                        continue
                    if alpha_disp is not None and beta_disp is not None:
                        pred[jd] = _predict_edge_ms(alpha_disp, beta_disp, ep_rank, jd, bdisp)
                    else:
                        pred[jd] = float(bdisp)
                if not pred:
                    longest_j = 0
                else:
                    longest_j = int(max(pred.items(), key=lambda kv: kv[1])[0])
                # Warmup (always executed)
                seed_base = cfg.routing.random.seed + M + ep_rank + longest_j * 13 + attempts
                for w in range(max(0, warmup)):
                    _ = measure_once(transformer_cfg, rm_try, cfg.model.hidden_size, bh, bp,
                                      rng_seed=seed_base + 997 * (w + 1))
                # Measure (always executed)
                key = (ep_rank, longest_j)
                pre_sums: Dict[int, float] = defaultdict(float)
                disp_sum = 0.0
                comb_sum = 0.0
                cnt = 0
                for t_idx in range(max(1, iters)):
                    t = measure_once(transformer_cfg, rm_try, cfg.model.hidden_size, bh, bp,
                                     rng_seed=seed_base + 1231 * (t_idx + 1))
                    for r, ms in t.preprocess_ms_per_rank.items():
                        pre_sums[r] += ms
                    if key in t.dispatch_ms_per_edge and t.dispatch_ms_per_edge[key]:
                        disp_sum += t.dispatch_ms_per_edge[key][0][1]
                    elif t.dispatch_ms_per_edge:
                        disp_sum += list(t.dispatch_ms_per_edge.values())[0][0][1]
                    if key in t.combine_ms_per_edge and t.combine_ms_per_edge[key]:
                        comb_sum += t.combine_ms_per_edge[key][0][1]
                    elif t.combine_ms_per_edge:
                        comb_sum += list(t.combine_ms_per_edge.values())[0][0][1]
                    cnt += 1
                # Record only if we still need this edge
                if longest_j in remaining_js:
                    for r, s in pre_sums.items():
                        preprocess_lat_ms[r].append(s / max(1, cnt))
                    bdisp, bcomb = edge_bytes.get(longest_j, (0, 0))
                    disp_meas[key].append((bdisp, disp_sum / max(1, cnt)))
                    comb_meas[key].append((bcomb, comb_sum / max(1, cnt)))
                    remaining_js.discard(longest_j)

    elif scenario == "coarse_baseline":
        # Coarse baseline: reuse baseline's serial per-edge traversal, but record a single
        # total time per edge on sender: preprocess_local + dispatch_total + combine_total.
        warmup = int(getattr(cfg.measure, 'warmup_iters', 0))
        iters = int(getattr(cfg.measure, 'iters', 1))

        def _targets_for_src(i: int) -> List[int]:
            if cfg.measure.baseline_target_dst is not None:
                return [int(cfg.measure.baseline_target_dst)]
            if getattr(cfg.measure, 'baseline_only_lower_triangle', False):
                return [j for j in range(ep_ws) if i > j]
            else:
                return [j for j in range(ep_ws) if j != i]

        # Build and broadcast edge list
        pairs: List[Tuple[int, int]] = []
        if dist.get_rank() == 0:
            for i in range(ep_ws):
                for j in _targets_for_src(i):
                    pairs.append((i, j))
        obj = [pairs]
        try:
            dist.broadcast_object_list(obj, src=0)
            pairs = obj[0]
        except Exception:
            pairs = [(i, j) for i in range(ep_ws) for j in _targets_for_src(i)]

        for M in cfg.measure.sizes:
            for (src, dst) in pairs:
                try:
                    dist.barrier()
                except Exception:
                    pass
                rm_edge = _build_baseline_map_for_edge(ep_ws, cfg.model.num_experts, cfg.model.topk,
                                                       src_rank=src, dst_rank=dst, M_tokens=M,
                                                       num_local_experts=num_local)
                seed_base = cfg.routing.random.seed + M + src * 131 + dst * 17
                # Warmup
                for w in range(max(0, warmup)):
                    _ = measure_once(transformer_cfg, rm_edge, cfg.model.hidden_size, bh, bp,
                                      rng_seed=seed_base + 997 * (w + 1))
                # Measure and average only on sender (use single CUDA window over preprocess+dispatch+combine)
                coarse_sum = 0.0
                cnt = 0
                for t_idx in range(max(1, iters)):
                    t = measure_once(transformer_cfg, rm_edge, cfg.model.hidden_size, bh, bp,
                                     rng_seed=seed_base + 1231 * (t_idx + 1))
                    if ep_rank == src:
                        total_cuda = float(getattr(t, 'total_cuda_ms', 0.0))
                        # Fallback to host sum if CUDA total is unavailable
                        if total_cuda <= 0.0:
                            pre_local = 0.0
                            if t.preprocess_ms_per_rank:
                                pre_local = float(list(t.preprocess_ms_per_rank.values())[0])
                            total_cuda = pre_local + float(t.dispatch_total_ms) + float(t.combine_total_ms)
                        coarse_sum += total_cuda
                    cnt += 1
                if ep_rank == src:
                    M_assign = M * cfg.model.topk
                    bcoarse = bytes_for_combine(M_assign, cfg.model.hidden_size, bh)  # ignore probs per requirement
                    local_coarse_total_ms_per_edge[(src, dst)].append((float(bcoarse), coarse_sum / max(1, cnt)))
                try:
                    dist.barrier()
                except Exception:
                    pass

    elif scenario == "a2a_validation":
        if cfg.routing.mode != "custom" or not cfg.routing.custom.csv_path:
            raise ValueError("a2a_validation requires routing.mode=custom with a valid CSV path")
        rm = build_custom_routing_map_from_csv(cfg.routing.custom.csv_path, ep_ws, cfg.model.num_experts, cfg.model.topk)
        rm = materialize_custom_placeholders(rm, num_local_experts=num_local)
        # One measurement per size (sizes can be used to repeat runs)
        warmup = int(getattr(cfg.measure, 'warmup_iters', 0))
        iters = int(getattr(cfg.measure, 'iters', 1))
        for M in cfg.measure.sizes:
            seed_base = cfg.routing.random.seed + M + ep_rank
            # Warmup
            for w in range(max(0, warmup)):
                _ = measure_once(transformer_cfg, rm, cfg.model.hidden_size, bh, bp, rng_seed=seed_base + 997 * (w + 1))
            # Measure and attribute to the longest edge of this src
            pre_sums: Dict[int, float] = defaultdict(float)
            edge_bytes = _edge_bytes_from_map(rm, cfg.model.hidden_size, bh, bp, num_local, ep_rank)
            if edge_bytes:
                longest_j = max(edge_bytes.items(), key=lambda kv: kv[1][0])[0]
                bdisp, bcomb = edge_bytes[longest_j]
                key = (ep_rank, longest_j)
                disp_sum = 0.0
                comb_sum = 0.0
                cnt = 0
                for t_idx in range(max(1, iters)):
                    t = measure_once(transformer_cfg, rm, cfg.model.hidden_size, bh, bp, rng_seed=seed_base + 1231 * (t_idx + 1))
                    for r, ms in t.preprocess_ms_per_rank.items():
                        pre_sums[r] += ms
                    if key in t.dispatch_ms_per_edge and t.dispatch_ms_per_edge[key]:
                        disp_sum += t.dispatch_ms_per_edge[key][0][1]
                    if key in t.combine_ms_per_edge and t.combine_ms_per_edge[key]:
                        comb_sum += t.combine_ms_per_edge[key][0][1]
                    cnt += 1
                for r, s in pre_sums.items():
                    preprocess_lat_ms[r].append(s / max(1, cnt))
        disp_meas[key].append((bdisp, disp_sum / max(1, cnt)))
        comb_meas[key].append((bcomb, comb_sum / max(1, cnt)))

    else:
        # Uniform all-to-all: each rank sends equal assignments to every other rank.
        # Build one map per M and measure; attribute bytes per edge according to uniform counts.
        if scenario == "uniform_all2all":
            warmup = int(getattr(cfg.measure, 'warmup_iters', 0))
            iters = int(getattr(cfg.measure, 'iters', 1))
            exclude_self = not bool(getattr(cfg.measure, 'include_self_edge', False))
            for M in cfg.measure.sizes:
                rm = build_uniform_routing_map(ep_ws, cfg.model.num_experts, cfg.model.topk,
                                               tokens_per_rank=M, exclude_self=exclude_self)
                seed_base = cfg.routing.random.seed + M + ep_rank
                # Warmup
                for w in range(max(0, warmup)):
                    _ = measure_once(transformer_cfg, rm, cfg.model.hidden_size, bh, bp,
                                      rng_seed=seed_base + 997 * (w + 1))
                # Measure and average
                pre_sums: Dict[int, float] = defaultdict(float)
                disp_sum = 0.0
                comb_sum = 0.0
                cnt = 0
                for t_idx in range(max(1, iters)):
                    t = measure_once(transformer_cfg, rm, cfg.model.hidden_size, bh, bp,
                                     rng_seed=seed_base + 1231 * (t_idx + 1))
                    for r, ms in t.preprocess_ms_per_rank.items():
                        pre_sums[r] += ms
                    # Any edge's dispatch/combine time equals the stage e2e on this map
                    # We will attribute it uniformly to all allowed destinations below.
                    if t.dispatch_ms_per_edge:
                        disp_sum += list(t.dispatch_ms_per_edge.values())[0][0][1]
                    if t.combine_ms_per_edge:
                        comb_sum += list(t.combine_ms_per_edge.values())[0][0][1]
                    cnt += 1
                for r, s in pre_sums.items():
                    preprocess_lat_ms[r].append(s / max(1, cnt))
                # Attribute bytes uniformly across destinations
                num_dests = ep_ws - (1 if exclude_self else 0)
                total_assign = M * cfg.model.topk
                base = total_assign // num_dests
                rem = total_assign % num_dests
                dests = [j for j in range(ep_ws) if (j != ep_rank) or not exclude_self]
                # Deterministic order
                for i, j in enumerate(dests):
                    M_assign = base + (1 if i < rem else 0)
                    bdisp = bytes_for_dispatch(M_assign, cfg.model.hidden_size, bh, bp)
                    bcomb = bytes_for_combine(M_assign, cfg.model.hidden_size, bh)
                    key = (ep_rank, j)
                    disp_meas[key].append((bdisp, disp_sum / max(1, cnt)))
                    comb_meas[key].append((bcomb, comb_sum / max(1, cnt)))
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    # Write per-rank measurements
    meas_path = os.path.join(meas_dir_ts, f"meas_rank{ep_rank}.json")
    dump_json({
        "run_id": run_id,
        "timestamp_unix": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "preprocess_ms_per_rank": {str(k): v for k, v in preprocess_lat_ms.items()},
        "dispatch_ms_per_edge": {f"{i}-{j}": pairs for (i,j), pairs in disp_meas.items()},
        "combine_ms_per_edge": {f"{i}-{j}": pairs for (i,j), pairs in comb_meas.items()},
        # Local per-rank timers (averaged per (M, src, dst) measurement)
        "local_rank_measurements": local_records,
        # Optional coarse baseline per-edge total pairs for this rank
        "coarse_total_ms_per_edge": {f"{i}-{j}": pairs for (i,j), pairs in local_coarse_total_ms_per_edge.items()},
    }, meas_path)

    # Fit + plots on rank 0
    dist.barrier()
    if ep_rank == 0 and cfg.fit.enable:
        all_disp: Dict[Tuple[int,int], Dict[str, List[Tuple[float,float]]]] = defaultdict(lambda: {"dispatch": [], "combine": []})
        all_comb: Dict[Tuple[int,int], Dict[str, List[Tuple[float,float]]]] = defaultdict(lambda: {"dispatch": [], "combine": []})
        all_pre: Dict[int, List[float]] = defaultdict(list)
        # Per-rank local timers aggregation
        local_rank_measurements: Dict[int, List[dict]] = defaultdict(list)
        # Per-rank per-edge measurement collectors (for per-rank fits)
        rank_disp_meas: Dict[int, Dict[Tuple[int,int], Dict[str, List[Tuple[float,float]]]]] = {
            r: defaultdict(lambda: {"dispatch": []}) for r in range(ep_ws)
        }
        rank_comb_meas: Dict[int, Dict[Tuple[int,int], Dict[str, List[Tuple[float,float]]]]] = {
            r: defaultdict(lambda: {"combine": []}) for r in range(ep_ws)
        }

        # Coarse pairs collectors (global and per rank)
        coarse_all_total_by_edge: Dict[Tuple[int,int], List[Tuple[float,float]]] = defaultdict(list)
        rank_coarse_total_by_edge: Dict[int, Dict[Tuple[int,int], List[Tuple[float,float]]]] = {
            r: defaultdict(list) for r in range(ep_ws)
        }

        for r in range(ep_ws):
            path = os.path.join(meas_dir_ts, f"meas_rank{r}.json")
            with open(path, "r") as f:
                d = json.load(f)
            for rk, arr in d["preprocess_ms_per_rank"].items():
                for ms in arr:
                    all_pre[int(rk)].append(ms)
            # Collect local per-rank timers if present
            for rec in d.get("local_rank_measurements", []):
                try:
                    # Ensure minimal schema
                    local_rank_measurements[int(r)].append({
                        "M": int(rec.get("M", 0)),
                        "src": int(rec.get("src", -1)),
                        "dst": int(rec.get("dst", -1)),
                        "dispatch_ms": float(rec.get("dispatch_ms", 0.0)),
                        "combine_ms": float(rec.get("combine_ms", 0.0)),
                    })
                except Exception:
                    pass
            # Coarse total pairs per edge (if any)
            def _parse_edge_pairs(dic):
                out = {}
                for k, v in (dic or {}).items():
                    i, j = map(int, k.split("-"))
                    out[(i,j)] = [(float(b), float(ms)) for (b, ms) in v]
                return out
            coarse_dict = _parse_edge_pairs(d.get("coarse_total_ms_per_edge"))
            for edge, pairs in coarse_dict.items():
                coarse_all_total_by_edge[edge].extend(pairs)
                rank_coarse_total_by_edge[int(r)][edge].extend(pairs)
            def parse_edges(dic):
                out = {}
                for k, v in dic.items():
                    i, j = map(int, k.split("-"))
                    out[(i,j)] = v
                return out
            d_disp = parse_edges(d["dispatch_ms_per_edge"])
            d_comb = parse_edges(d["combine_ms_per_edge"])
            for edge, pairs in d_disp.items():
                all_disp[edge]["dispatch"].extend(pairs)
                # Also attribute to the file rank r for per-rank fit
                rank_disp_meas[int(r)][edge]["dispatch"].extend(pairs)
            for edge, pairs in d_comb.items():
                all_comb[edge]["combine"].extend(pairs)
                rank_comb_meas[int(r)][edge]["combine"].extend(pairs)

        # Fit per edge (global) using coarse-total if provided (as separate section), otherwise use stage edges
        coarse_fit_global = None
        if coarse_all_total_by_edge:
            meas_total = defaultdict(lambda: {"dispatch": []})
            for edge, pairs in coarse_all_total_by_edge.items():
                meas_total[edge]["dispatch"].extend(pairs)
            ct_alpha, ct_beta, ct_r2 = grid_fit_per_edge(meas_total, "dispatch")
            coarse_fit_global = (ct_alpha, ct_beta, ct_r2)
        if not coarse_fit_global:
            alpha_disp, beta_disp, r2_disp = grid_fit_per_edge(all_disp, "dispatch")
            alpha_comb, beta_comb, r2_comb = grid_fit_per_edge(all_comb, "combine")

        # Preprocess per-rank avg
        pre_lat_avg = [[0.0 for _ in range(ep_ws)]]
        pre_list = [sum(v)/max(1,len(v)) for r, v in sorted(all_pre.items(), key=lambda x: x[0])]
        if pre_list:
            pre_lat_avg = [pre_list]

        results = {}
        if coarse_fit_global:
            ct_alpha, ct_beta, ct_r2 = coarse_fit_global
            # Put coarse results under a dedicated section
            results["coarse_total"] = {
                "alpha_ms": ct_alpha,
                "beta_ms_per_byte": ct_beta,
                "r_squared": ct_r2,
            }
        # Also add stage fits if any (may be zero matrices in coarse-only runs)
        results["dispatch"] = {
            "alpha_ms": locals().get("alpha_disp", []),
            "beta_ms_per_byte": locals().get("beta_disp", []),
            "r_squared": locals().get("r2_disp", []),
        }
        results["combine"] = {
            "alpha_ms": locals().get("alpha_comb", []),
            "beta_ms_per_byte": locals().get("beta_comb", []),
            "r_squared": locals().get("r2_comb", []),
        }
        results["preprocess"] = {"latency_ms": pre_lat_avg}
        res_path = os.path.join(results_dir_ts, "fit_results.json")
        results_with_meta = {
            "run_id": run_id,
            "timestamp_unix": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            **results,
        }
        dump_json(results_with_meta, res_path)

        # Per-rank fits written to results as separate files (fit_results_rank{r}.json)
        # Using only that rank's own per-edge timing pairs
        # Also include this rank's preprocess avg for reference
        for r in range(ep_ws):
            ct_rank = None
            if rank_coarse_total_by_edge[r]:
                meas_r = defaultdict(lambda: {"dispatch": []})
                for edge, pairs in rank_coarse_total_by_edge[r].items():
                    meas_r[edge]["dispatch"].extend(pairs)
                a_ct_r, b_ct_r, r2_ct_r = grid_fit_per_edge(meas_r, "dispatch")
                ct_rank = (a_ct_r, b_ct_r, r2_ct_r)
            # Stage fits per rank (if any)
            a_d_r, b_d_r, r2_d_r = grid_fit_per_edge(rank_disp_meas[r], "dispatch")
            a_c_r, b_c_r, r2_c_r = grid_fit_per_edge(rank_comb_meas[r], "combine")
            pre_avg_r = 0.0
            if all_pre.get(r):
                pre_avg_r = sum(all_pre[r]) / max(1, len(all_pre[r]))
            per_rank_results = {"rank": int(r), "preprocess": {"latency_ms": [[pre_avg_r]]}}
            if ct_rank:
                a_ct_r, b_ct_r, r2_ct_r = ct_rank
                per_rank_results["coarse_total"] = {
                    "alpha_ms": a_ct_r,
                    "beta_ms_per_byte": b_ct_r,
                    "r_squared": r2_ct_r,
                }
            per_rank_results["dispatch"] = {
                "alpha_ms": a_d_r,
                "beta_ms_per_byte": b_d_r,
                "r_squared": r2_d_r,
            }
            per_rank_results["combine"] = {
                "alpha_ms": a_c_r,
                "beta_ms_per_byte": b_c_r,
                "r_squared": r2_c_r,
            }
            out_path = os.path.join(results_dir_ts, f"fit_results_rank{int(r)}.json")
            dump_json({
                "run_id": run_id,
                "timestamp_unix": time.time(),
                "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
                **per_rank_results,
            }, out_path)

        # Also write per-rank local timers into measurement directory (keep results only for fits)
        for rk, recs in local_rank_measurements.items():
            try:
                summary = {
                    "count": len(recs),
                    "avg_dispatch_ms": (sum(x.get("dispatch_ms", 0.0) for x in recs) / max(1, len(recs))) if recs else 0.0,
                    "avg_combine_ms": (sum(x.get("combine_ms", 0.0) for x in recs) / max(1, len(recs))) if recs else 0.0,
                }
                per_rank_out = {
                    "run_id": run_id,
                    "rank": int(rk),
                    "timestamp_unix": time.time(),
                    "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
                    "local_rank_measurements": recs,
                    "summary": summary,
                }
                out_path = os.path.join(meas_dir_ts, f"local_rank{int(rk)}.json")
                dump_json(per_rank_out, out_path)
            except Exception:
                pass

        # Plots
        # Temporarily override plots dir via local variable
        old_plots_dir = cfg.paths.plots_dir
        try:
            cfg.paths.plots_dir = plots_dir_ts
            # Ensure stage fit matrices exist in coarse-only runs
            if 'alpha_disp' not in locals() or 'beta_disp' not in locals():
                alpha_disp = [[0.0 for _ in range(ep_ws)] for __ in range(ep_ws)]
                beta_disp = [[0.0 for _ in range(ep_ws)] for __ in range(ep_ws)]
            if 'alpha_comb' not in locals() or 'beta_comb' not in locals():
                alpha_comb = [[0.0 for _ in range(ep_ws)] for __ in range(ep_ws)]
                beta_comb = [[0.0 for _ in range(ep_ws)] for __ in range(ep_ws)]
            _maybe_plot(cfg, ep_ws, all_pre, all_disp, all_comb, alpha_disp, beta_disp, alpha_comb, beta_comb)
        finally:
            cfg.paths.plots_dir = old_plots_dir

    if ep_rank == 0:
        print("Run ID:", run_id)
        print("Done. Measurements saved to:", meas_dir_ts)
        if cfg.fit.enable:
            print("Fit results saved to:", os.path.join(results_dir_ts, "fit_results.json"))
        if cfg.plots.enable:
            print("Plots saved to:", plots_dir_ts)

if __name__ == "__main__":
    main()
