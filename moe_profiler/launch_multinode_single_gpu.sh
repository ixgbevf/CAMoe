#!/usr/bin/env bash

# Simple multi-node, single-GPU-per-node launcher for tools/moe_profiler
# Works with torchrun static master addressing (no etcd needed).
#
# Usage (on every node, with node-specific NODE_RANK):
#   NNODES=4 NODE_RANK=0 MASTER_ADDR=host0 MASTER_PORT=29500 \
#   bash tools/moe_profiler/launch_multinode_single_gpu.sh tools/moe_profiler/examples/baseline_random.yaml
#
# Note:
# - Ensure config.parallel.ep_size equals the total number of nodes (world size).
# - This script uses nproc_per_node=1, so each node runs 1 process bound to LOCAL_RANK=0 (GPU 0).
# - Set CUDA_VISIBLE_DEVICES if you want a different GPU per node.

set -euo pipefail

PYTHON=${PYTHON:-python}
CFG_PATH=${1:-tools/moe_profiler/examples/baseline_random.yaml}

# Torch distributed settings
NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NPROC_PER_NODE=1

# Optional: pin to a single visible GPU per node (defaults to device 0)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# NCCL and perf related suggestions (safe defaults; adjust as needed)
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# Set network interface if needed, e.g. eth0 or ib0
# export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}

# Repo root and PYTHONPATH for megatron.core
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_DIR=$( cd -- "${SCRIPT_DIR}/../.." &> /dev/null && pwd )
cd "${REPO_DIR}"

if [[ ! -f "${CFG_PATH}" ]]; then
  echo "Config file not found: ${CFG_PATH}" >&2
  exit 1
fi

export PYTHONPATH=${PYTHONPATH:-}:${REPO_DIR}/megatron

echo "== MoE Profiler | multi-node single-GPU launcher =="
echo "Repo:        ${REPO_DIR}"
echo "Config:      ${CFG_PATH}"
echo "NNODES:      ${NNODES}"
echo "NODE_RANK:   ${NODE_RANK}"
echo "MASTER:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "GPU/Node:    ${NPROC_PER_NODE} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "PYTHONPATH+= ${REPO_DIR}/megatron"
echo "Hint: set parallel.ep_size in config to ${NNODES} for EP-only parallel"
echo

exec torchrun \
  --nnodes "${NNODES}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  tools/moe_profiler/main.py "${CFG_PATH}"

