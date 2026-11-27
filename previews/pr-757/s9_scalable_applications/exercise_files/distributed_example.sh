#!/bin/bash

# this example uses a single node (`NUM_NODES=1`) w/ 4 GPUs (`NUM_GPUS_PER_NODE=4`)
export NUM_NODES=1
export NUM_GPUS_PER_NODE=4
export NODE_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

# launch your script w/ `torchrun` (add any arguments to your script at the end, e.g. `--seed 42`)
torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    distributed_example.py
