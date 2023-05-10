import torch
import torch.nn as nn

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, input):
        p = torch.softmax(input, dim=1)
        log_p = torch.log_softmax(input, dim=1)
        loss = -torch.sum(p * log_p, dim=1).mean()
        return loss