import torch
from torch import nn
import torch.nn.functional as F


class KLregular(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q_pred, P_targ):
        return torch.mean(torch.sum(P_targ * torch.log2(P_targ / Q_pred), dim=1))
        # return torch.sum(P_targ * torch.log2(P_targ / Q_pred))


class KLinverse(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q_pred, P_targ):
        return torch.mean(torch.sum(Q_pred * torch.log2(Q_pred / P_targ), dim=1))
        # return torch.sum(Q_pred * torch.log2(Q_pred / P_targ))


class CrossEntropySoft(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q_pred, P_targ):
        return torch.mean(-torch.sum(P_targ * torch.log2(Q_pred), dim=1))
        # return -torch.sum(P_targ * torch.log2(Q_pred))


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q_pred, P_targ):
        return torch.mean(-torch.sum(P_targ * torch.log2(Q_pred), dim=1))
