#!/usr/bin/env python
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optimizer
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import func

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

n_gpus = torch.cuda.device_count()
#assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
world_size = n_gpus
run_demo(func.demo_basic, world_size)
