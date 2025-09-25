\
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class FitConfig:
    enable: bool = True

@dataclass
class PlotsConfig:
    enable: bool = True

@dataclass
class RoutingRandomConfig:
    tokens_per_rank: int = 4096
    topk: int = 2
    seed: int = 17
    jitter_ratio: float = 0.1  # max relative difference across ranks for outgoing tokens
    bias_prob: float = 0.7     # probability to direct a token to the target dst when biasing

@dataclass
class RoutingCustomConfig:
    csv_path: Optional[str] = None  # CSV schema: src_rank,dst_ep_rank,num_tokens[,topk_slot]
    # If topk>1, you can repeat rows for multiple slots or let generator split evenly.

@dataclass
class RoutingConfig:
    mode: str = "random"  # "random" | "custom"
    random: RoutingRandomConfig = field(default_factory=RoutingRandomConfig)
    custom: RoutingCustomConfig = field(default_factory=RoutingCustomConfig)

@dataclass
class ModelConfig:
    hidden_size: int = 4096
    num_experts: int = 8
    topk: int = 2
    bh: int = 2   # bytes per hidden element (e.g., bf16/fp16=2, fp32=4)
    bp: int = 2   # bytes per prob element

@dataclass
class ParallelConfig:
    ep_size: Optional[int] = None  # if None, infer from env / parallel_state
    tp_size: int = 1
    pp_size: int = 1

@dataclass
class MeasureConfig:
    sizes: List[int] = field(default_factory=lambda: [1024, 2048, 4096, 8192])
    warmup_iters: int = 5
    iters: int = 20
    max_attempts: int = 50  # for all2all_profile edge selection
    sync_cuda: bool = True
    # Baseline options
    baseline_only_lower_triangle: bool = True  # measure only edges with src_rank > dst_rank
    baseline_target_dst: Optional[int] = None  # if set, measure only this dst for current src
    # Uniform all2all options
    include_self_edge: bool = False  # whether to include i==j traffic in uniform_all2all
    # Dispatch measurement side: 'sender' (default), 'receiver', or 'both'.
    # In baseline mode, 'receiver' records dispatch time on destination rank j for edge (i->j).
    dispatch_side: str = "sender"
    # Combine timing mode:
    #  - 'receiver_local': use receiver rank's local end-to-end window (default legacy behavior)
    #  - 'cross_rank_diff': use cross-end time difference with synchronized clocks
    #  - 'max_of_locals': use max(sender_local, receiver_local)
    combine_timing_mode: str = "receiver_local"
    # Direction for cross_rank_diff:
    #  - 'sender_to_receiver': end(receiver) - start(sender)
    #  - 'receiver_to_sender': end(sender) - start(receiver)
    combine_diff_direction: str = "sender_to_receiver"
    # When true, compute combine time on sender as (dispatch+combine total) - dispatch, all via local CUDA timing.
    combine_from_sender_diff: bool = False

@dataclass
class SceneConfig:
    scenario: str = "baseline"  # "baseline" | "all2all_profile" | "a2a_validation"

@dataclass
class PathsConfig:
    data_dir: str = "data"
    measurements_dir: str = "measurements"
    results_dir: str = "results"
    plots_dir: str = "plots"
    # Optional: directory (or file path) to load baseline fit_results.json from
    baseline_results_dir: Optional[str] = None

@dataclass
class ProfilerConfig:
    scene: SceneConfig = field(default_factory=SceneConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    plots: PlotsConfig = field(default_factory=PlotsConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    measure: MeasureConfig = field(default_factory=MeasureConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    log_level: str = "INFO"

def from_dict(d: Dict[str, Any]) -> "ProfilerConfig":
    # Very lightweight nested conversion with defaults
    def _get(dct, key, typ, default):
        return typ(**dct.get(key, {})) if isinstance(dct.get(key, {}), dict) else default

    scene = _get(d, "scene", SceneConfig, SceneConfig())
    fit = _get(d, "fit", FitConfig, FitConfig())
    plots = _get(d, "plots", PlotsConfig, PlotsConfig())
    routing_random = _get(d.get("routing", {}), "random", RoutingRandomConfig, RoutingRandomConfig())
    routing_custom = _get(d.get("routing", {}), "custom", RoutingCustomConfig, RoutingCustomConfig())
    routing = RoutingConfig(mode=d.get("routing", {}).get("mode", "random"),
                            random=routing_random, custom=routing_custom)
    model = _get(d, "model", ModelConfig, ModelConfig())
    parallel = _get(d, "parallel", ParallelConfig, ParallelConfig())
    measure = _get(d, "measure", MeasureConfig, MeasureConfig())
    paths = _get(d, "paths", PathsConfig, PathsConfig())
    log_level = d.get("log_level", "INFO")
    return ProfilerConfig(scene=scene, fit=fit, plots=plots, routing=routing, model=model,
                          parallel=parallel, measure=measure, paths=paths, log_level=log_level)
