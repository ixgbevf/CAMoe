\
from __future__ import annotations
import os, time, inspect
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch
import torch.distributed as dist
import time

from megatron.core import parallel_state
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher

from .routing_map import RoutingMap, materialize_custom_placeholders
from .data import derive_hidden_and_probs_for_rank
from .utils import get_ep_size_and_rank, ensure_cuda_device_from_env, bytes_for_dispatch, bytes_for_combine

@dataclass
class Timings:
    preprocess_ms_per_rank: Dict[int, float]  # rank -> ms
    dispatch_ms_per_edge: Dict[Tuple[int,int], List[Tuple[int, float]]]  # (src,dst) -> list of (bytes, ms)
    combine_ms_per_edge: Dict[Tuple[int,int], List[Tuple[int, float]]]
    # Rank-local total timings for the whole dispatch/combine stage windows (for receiver-side attribution in baseline)
    dispatch_total_ms: float
    combine_total_ms: float
    # Host wall-clock timestamps (ms) for combine window, used for cross-rank diff timing
    combine_host_t_start_ms: float
    combine_host_t_end_ms: float
    # CUDA event timings
    dispatch_cuda_ms: float
    combine_cuda_ms: float
    total_cuda_ms: float

def _init_dist(tp: int, pp: int, ep: Optional[int]) -> None:
    # Initialize torch.distributed and Megatron Core parallel state
    if not dist.is_initialized():
        backend = "nccl"
        dist.init_process_group(backend=backend)
    ensure_cuda_device_from_env()
    # initialize model parallel
    # signature compatible init
    init_fn = parallel_state.initialize_model_parallel
    try:
        sig = inspect.signature(init_fn)
        kwargs = {}
        if 'tensor_model_parallel_size' in sig.parameters:
            kwargs['tensor_model_parallel_size'] = tp
        if 'pipeline_model_parallel_size' in sig.parameters:
            kwargs['pipeline_model_parallel_size'] = pp
        if 'expert_model_parallel_size' in sig.parameters and ep is not None:
            kwargs['expert_model_parallel_size'] = ep
        init_fn(**kwargs)
    except Exception:
        # fallback older signature
        try:
            parallel_state.initialize_model_parallel(tp, pp)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model parallel: {e}")

def _build_dispatcher(transformer_cfg: TransformerConfig, num_local_experts: int, local_expert_indices: List[int]) -> MoEAlltoAllTokenDispatcher:
    # Build required process groups for the dispatcher from current parallel state
    model_pgs = ModelCommProcessGroups.use_mpu_process_groups(required_pgs=['ep', 'expt_tp', 'tp_ep'])
    try:
        return MoEAlltoAllTokenDispatcher(
            num_local_experts=num_local_experts,
            local_expert_indices=local_expert_indices,
            config=transformer_cfg,
            model_comm_pgs=model_pgs,
        )
    except TypeError:
        # Older signature fallback without named args
        return MoEAlltoAllTokenDispatcher(num_local_experts, local_expert_indices, transformer_cfg, model_pgs)  # type: ignore

def _compute_num_local_and_indices(num_experts: int) -> Tuple[int, int, List[int]]:
    ep_ws, ep_rank = get_ep_size_and_rank(None)
    assert num_experts % ep_ws == 0, f"num_experts={num_experts} must be divisible by EP world size={ep_ws}"
    num_local = num_experts // ep_ws
    local_indices = [ep_rank * num_local + i for i in range(num_local)]
    return num_local, ep_rank, local_indices

def _cuda_ms() -> float:
    torch.cuda.synchronize()
    return time.perf_counter() * 1000.0

def measure_once(
    transformer_cfg: TransformerConfig,
    rm: RoutingMap,
    hidden_size: int,
    bytes_bh: int,
    bytes_bp: int,
    rng_seed: int,
) -> Timings:
    """Measure preprocess, dispatch(token_permutation), combine(token_unpermutation)."""
    num_local, ep_rank, local_indices = _compute_num_local_and_indices(transformer_cfg.num_moe_experts)
    # Materialize custom placeholders to concrete expert ids
    if any((rm.per_rank_indices[r].numel() > 0 and (rm.per_rank_indices[r] < 0).any().item()) for r in rm.per_rank_indices):
        rm = materialize_custom_placeholders(rm, num_local)

    dispatcher = _build_dispatcher(transformer_cfg, num_local, local_indices)

    # This rank's routing
    routing_indices = rm.per_rank_indices.get(ep_rank, torch.zeros((0, rm.topk), dtype=torch.long))
    # Derive hidden/probs (on CUDA)
    hidden_states, probs = derive_hidden_and_probs_for_rank(routing_indices, hidden_size, dtype=torch.bfloat16, seed=rng_seed)

    # Build routing_map (bool [num_tokens, num_experts]) and full probs ([num_tokens, num_experts])
    num_tokens = routing_indices.size(0)
    num_experts = transformer_cfg.num_moe_experts or 0
    device = hidden_states.device
    routing_map_bool = torch.zeros((num_tokens, num_experts), dtype=torch.bool, device=device)
    probs_full = torch.zeros((num_tokens, num_experts), dtype=torch.float32, device=device)
    if num_tokens > 0:
        idx_rows = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, routing_indices.size(1))
        routing_map_bool[idx_rows, routing_indices.to(device=device, dtype=torch.long)] = True
        probs_full[idx_rows, routing_indices.to(device=device, dtype=torch.long)] = probs.to(device=device, dtype=torch.float32)

    # --- preprocess (metadata only: CPU/GPU sync and splits) ---
    preprocess_ms: Dict[int, float] = {}
    torch.cuda.synchronize()
    # Record a single CUDA event window that spans preprocess + dispatch + combine
    e_total_start = torch.cuda.Event(enable_timing=True)
    e_total_end = torch.cuda.Event(enable_timing=True)
    e_total_start.record()
    tpre0 = _cuda_ms()
    _ = dispatcher.preprocess(routing_map_bool)
    tpre1 = _cuda_ms()
    preprocess_ms[ep_rank] = float(tpre1 - tpre0)

    # --- dispatch end-to-end (dispatch_preprocess + token_dispatch + dispatch_postprocess) ---
    torch.cuda.synchronize()
    host_t0_ms = time.time() * 1000.0
    t0 = _cuda_ms()
    e_disp_start = torch.cuda.Event(enable_timing=True)
    e_disp_end = torch.cuda.Event(enable_timing=True)
    e_disp_start.record()
    perm_inp, perm_probs = dispatcher.dispatch_preprocess(hidden_states, routing_map_bool, probs_full)
    global_inp, global_probs = dispatcher.token_dispatch(perm_inp, perm_probs)
    dispatched_input, tokens_per_expert, _ = dispatcher.dispatch_postprocess(global_inp, global_probs)
    e_disp_end.record()
    torch.cuda.synchronize()
    t1 = _cuda_ms()
    dispatch_ms = float(t1 - t0)
    dispatch_ms_cuda = float(e_disp_start.elapsed_time(e_disp_end))

    # Approximate bytes per edge for dispatch: need to count token assignments towards each dst EP rank
    # For this rank, routing_indices contains global expert ids; derive dst rank per slot.
    dst_ranks = (routing_indices // num_local) if routing_indices.numel() > 0 else torch.zeros((0, rm.topk), dtype=torch.long, device="cpu")
    # Build edge counts: M_ij = number of assignments from src=ep_rank to dst=j
    dispatch_counts_per_dst: Dict[int, int] = {}
    if routing_indices.numel() > 0:
        dst_cpu = dst_ranks.detach().cpu().numpy().reshape(-1)
        for j in dst_cpu.tolist():
            dispatch_counts_per_dst[int(j)] = dispatch_counts_per_dst.get(int(j), 0) + 1

    # --- combine (preprocess + alltoall + postprocess) ---
    # Simulate expert output as identity tensor (same shape)
    expert_output = dispatched_input
    torch.cuda.synchronize()
    t0 = _cuda_ms()
    e_comb_start = torch.cuda.Event(enable_timing=True)
    e_comb_end = torch.cuda.Event(enable_timing=True)
    e_comb_start.record()
    cprep = dispatcher.combine_preprocess(expert_output)
    cglobal = dispatcher.token_combine(cprep)
    _out = dispatcher.combine_postprocess(cglobal)
    e_comb_end.record()
    torch.cuda.synchronize()
    t1 = _cuda_ms()
    host_t1_ms = time.time() * 1000.0
    combine_ms = float(t1 - t0)
    combine_ms_cuda = float(e_comb_start.elapsed_time(e_comb_end))

    # Combine edge counts: tokens are returned to this src rank from all dst; symmetric to dispatch_counts_per_dst
    # For combine, the total bytes for edge (src=ep_rank, dst=j) equals number of assignments dispatched earlier to j.
    combine_counts_per_dst = dispatch_counts_per_dst

    # Build results structures
    disp_edge = {}
    comb_edge = {}
    for j, M in dispatch_counts_per_dst.items():
        b = bytes_for_dispatch(M, hidden_size, bytes_bh, bytes_bp)
        disp_edge[(ep_rank, int(j))] = [(b, dispatch_ms)]
    for j, M in combine_counts_per_dst.items():
        b = bytes_for_combine(M, hidden_size, bytes_bh)
        comb_edge[(ep_rank, int(j))] = [(b, combine_ms)]

    # Record the total CUDA window end
    e_total_end.record()
    torch.cuda.synchronize()
    total_cuda_whole_ms = float(e_total_start.elapsed_time(e_total_end))

    return Timings(preprocess_ms_per_rank=preprocess_ms,
                   dispatch_ms_per_edge=disp_edge,
                   combine_ms_per_edge=comb_edge,
                   dispatch_total_ms=dispatch_ms,
                   combine_total_ms=combine_ms,
                   combine_host_t_start_ms=float(host_t0_ms),
                   combine_host_t_end_ms=float(host_t1_ms),
                   dispatch_cuda_ms=dispatch_ms_cuda,
                   combine_cuda_ms=combine_ms_cuda,
                   total_cuda_ms=total_cuda_whole_ms)
