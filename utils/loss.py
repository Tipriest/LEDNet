import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None, ignore_label=255):
        super(CrossEntropyLoss2d, self).__init__()

        self.loss = nn.NLLLoss(weight, ignore_index=ignore_label)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, dim=1), targets)






