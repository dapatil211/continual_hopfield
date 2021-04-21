import torch
import torch.nn as nn


def get_model(args):
    pass


class BaseModel(nn.Module):
    def switch_task(self):
        pass

    def get_loss(X, y, mask):
        pass

    def get_metrics(X, y, mask):
        pass
