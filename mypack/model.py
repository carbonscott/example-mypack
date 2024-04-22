import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.layer = nn.Linear(input_dim, output_dim)


    def forward(self, x):
        return self.layer(x)


    def forward_loss(self, x, y):
        x = self.layer(x)
        loss = F.mse_loss(x, y)
        return loss
