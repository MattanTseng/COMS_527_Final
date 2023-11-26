# This code follows the pytorch docs: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '12355'

    dist.init_process_group("gloo", rank = rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def training_basic(rank, world_size):
    setup(rank, world_size)

    my_network = DQN()



def run_training(trainin_fn, world_size):
    mp.spawn(trainin_fn, 
             args=(world_size,),
             nprocs=world_size, 
             join = True)
