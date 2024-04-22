import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tqdm
import os

from torch.nn.parallel import DistributedDataParallel as DDP

from mypack.model import MyNN
from mypack.data  import MyDataset

import torch.distributed as dist

from datetime import timedelta

# ----------------------------------------------------------------------- #
#  User params SETUP
# ----------------------------------------------------------------------- #
# [(224 x 224, 224 x 224), ()]
B, C, H, W = 1000, 1, 32, 32
dist_backend = 'nccl'
total_epoch = 10

# ----------------------------------------------------------------------- #
#  DIST SETUP
# ----------------------------------------------------------------------- #
# -- DIST init
# --- Initialize distributed environment
uses_dist = int(os.environ.get("RANK", -1)) != -1
if uses_dist:
    dist_rank       = int(os.environ["RANK"      ])
    dist_local_rank = int(os.environ["LOCAL_RANK"])
    dist_world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend     = dist_backend,
                            rank        = dist_rank,
                            world_size  = dist_world_size,
                            timeout     = timedelta(seconds=900),
                            init_method = "env://",)
    print(f"RANK:{dist_rank},LOCAL_RANK:{dist_local_rank},WORLD_SIZE:{dist_world_size}")
else:
    dist_rank       = 0
    dist_local_rank = 0
    dist_world_size = 1
    print(f"NO DIST is used.  RANK:{dist_rank},LOCAL_RANK:{dist_local_rank},WORLD_SIZE:{dist_world_size}")

# DEVICE
device = f'cuda:{dist_local_rank}' if torch.cuda.is_available() else 'cpu'
if device != 'cpu': torch.cuda.set_device(device)


# ----------------------------------------------------------------------- #
#  Model
# ----------------------------------------------------------------------- #
input_dim = C * H * W
output_dim = C * H * W
model = MyNN(input_dim, output_dim)

# Wrapping the model in DDP...
if uses_dist:
    # Wrap it up using DDP...
    model = DDP(model, device_ids = [dist_local_rank], find_unused_parameters=False)

    dist.barrier()

model.to(device)

# ----------------------------------------------------------------------- #
#  Dataset
# ----------------------------------------------------------------------- #
data_list = torch.rand(B, 2, C, H, W)  # shape: [1000, 2, 1, 224, 224]
dataset = MyDataset(data_list)
size_batch = 10
num_workers = 2

sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
dataloader = torch.utils.data.DataLoader(
    dataset,
    sampler     = sampler,
    shuffle     = False,
    pin_memory  = True,
    batch_size  = size_batch,
    num_workers = num_workers,
)

# ----------------------------------------------------------------------- #
#  Training loop
# ----------------------------------------------------------------------- #

for epoch in range(total_epoch):

    model.train()

    if uses_dist:
        # Shuffle the training examples...
        sampler.set_epoch(epoch)

    for batch_data, batch_label in tqdm.tqdm(dataloader):
        batch_data = batch_data.to(device)
        batch_label = batch_label.to(device)

        batch_loss = model.forward_loss(batch_data, batch_label)
        print(batch_loss)
