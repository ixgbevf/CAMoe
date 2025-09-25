\
from __future__ import annotations
import os
from typing import Dict, Tuple
import torch
from megatron.core import parallel_state

def get_ep_size_and_rank(config_ep_size: int | None) -> Tuple[int, int]:
    if hasattr(parallel_state, "get_expert_model_parallel_world_size"):
        ws = parallel_state.get_expert_model_parallel_world_size()
        r  = parallel_state.get_expert_model_parallel_rank()
        if ws is None or ws == 0:
            # fallback to config
            ws = config_ep_size or torch.distributed.get_world_size()
            r  = torch.distributed.get_rank()
        return int(ws), int(r)
    else:
        # fallback
        ws = config_ep_size or torch.distributed.get_world_size()
        r  = torch.distributed.get_rank()
        return int(ws), int(r)

def ensure_cuda_device_from_env() -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0)))
    torch.cuda.set_device(local_rank)
    return local_rank

def bytes_for_dispatch(M: int, H: int, bh: int, bp: int) -> int:
    return int(M) * (int(H) * int(bh) + int(bp))

def bytes_for_combine(M: int, H: int, bh: int) -> int:
    return int(M) * (int(H) * int(bh))
