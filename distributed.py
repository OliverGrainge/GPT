import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os 

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    print("init process group: rank: ", rank)
    try:
        print(f"init process group: rank: {rank}")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        print(f"finished init process group: rank: {rank}")
    except Exception as e:
        print(f"Error during DDP initialization for rank {rank}: {str(e)}")
        raise
    print("init pr")

def cleanup():
    dist.destroy_process_group()


    